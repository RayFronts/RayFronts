"""Server to run and visualize semantic mapping from a posed RGBD data source

The map server can be queried with images or text with a messaging_service
online or with a predefined query file.

Loads the configs/default.yaml as the root config but all options can be
overwridden by other configs or through the command line. Check hydra-configs
for more details.

"""

import random
import os
import logging
import time
import atexit
import inspect
import threading
import signal
from enum import Enum
from functools import partial
from typing_extensions import List
import json

import torch
import torchvision
import numpy as np
import hydra

from rayfronts import datasets, visualizers, image_encoders, mapping, utils

logger = logging.getLogger(__name__)

# Default confidence threshold for HTTP queries (raw cosine similarity)
# Scores above this are considered valid matches
DEFAULT_QUERY_THRESHOLD = 0.10


class MappingServer:
    """Server performing mapping on a stream of posed RGBD data. Can be queried.

    Attributes:
      status: Status enum signaling the current state of the server.
      cfg: Stores the mapping system configuration as is.
      dataset: Stores the dataset/datasource object.
      vis: Stores the visualizer used. May be None.
      encoder: Stores the encoder model used by the mapper.
      mapper: Stores the mapper used.
      messaging_service: Stores the messaging service used for online querying.
    """

    class Status(Enum):
        INIT = 0  # Server is initializing
        MAPPING = 1  # Server is actively mapping
        IDLE = 2  # Server has stopped mapping and awaits any new queries.
        CLOSING = 3  # Server is in the process of shutting down.
        CLOSED = 4  # Server has shutdown.

    @torch.inference_mode()
    def __init__(self, cfg):
        self.status = MappingServer.Status.INIT
        self._status_lock = threading.RLock()
        self._last_pose_rdf = None  # latest camera pose in RDF frame (4x4)

        self.cfg = cfg
        self.dataset: datasets.PosedRgbdDataset = hydra.utils.instantiate(cfg.dataset)

        intrinsics_3x3 = self.dataset.intrinsics_3x3
        if "vox_size" in cfg.mapping:
            base_point_size = cfg.mapping.vox_size / 2
        else:
            base_point_size = None

        self.vis: visualizers.Mapping3DVisualizer = None
        if "vis" in cfg and cfg.vis is not None:
            self.vis = hydra.utils.instantiate(
                cfg.vis, intrinsics_3x3=intrinsics_3x3, base_point_size=base_point_size
            )

        # Ugly way to check if the chosen mapper constructor needs an encoder.
        c = getattr(mapping, cfg.mapping._target_.split(".")[-1])
        init_encoder = "encoder" in inspect.signature(c.__init__).parameters.keys()
        init_encoder = init_encoder and "encoder" in cfg
        mapper_kwargs = dict()

        self.encoder: image_encoders.ImageEncoder = None
        self.feat_compressor = None
        if (
            "feat_compressor" in self.cfg.mapping
            and self.cfg.mapping.feat_compressor is not None
        ):
            self.feat_compressor = hydra.utils.instantiate(
                self.cfg.mapping.feat_compressor
            )

        if init_encoder:
            encoder_kwargs = dict()
            if (
                cfg.querying.text_query_mode is not None
                and "RadioEncoder" in cfg.encoder
                and cfg.encoder.lang_model is None
            ):
                raise ValueError(
                    "Radio encoder must have a language model if text "
                    "querying is enabled."
                )
            if "NARadioEncoder" in cfg.encoder._target_:
                encoder_kwargs["input_resolution"] = [
                    self.dataset.rgb_h,
                    self.dataset.rgb_w,
                ]
            if (
                hasattr(self.dataset, "cat_name_to_index")
                and "classes" in cfg.encoder
                and cfg.encoder.classes is None
            ):
                encoder_kwargs["classes"] = self.dataset.cat_index_to_name[1:]

            self.encoder = hydra.utils.instantiate(cfg.encoder, **encoder_kwargs)
            mapper_kwargs["encoder"] = self.encoder
            mapper_kwargs["feat_compressor"] = self.feat_compressor

        self.mapper: mapping.RGBDMapping = hydra.utils.instantiate(
            cfg.mapping,
            intrinsics_3x3=intrinsics_3x3,
            visualizer=self.vis,
            **mapper_kwargs,
        )

        # Dictionary mapping a label group name to a list of string labels.
        # In the case of a text query, the label is the query. In case of image
        # querying, the label is the image file name.
        self._queries_labels = None

        # Dictionary mapping a label group name to an NxD torch tensor where
        # N corresponds to the number of queries in that group and D is the
        # feature dimension. N must equal len(self.queries_labels[k]) for a group.
        self._queries_feats = None

        # History is used when mapper is idling such that only new queries are
        # visualized as opposed to visualizing the full query set everytime.
        # This also depends on compute_prob value.
        self._queries_labels_history = set()

        # Flag to track if the set of queries has been updated or not.
        self._queries_updated = False

        # Color map mapping query label to color.
        self._query_cmap = dict()

        self._query_lock = threading.RLock()

        if cfg.querying.query_file is not None:
            with open(cfg.querying.query_file, "r", encoding="UTF-8") as f:
                if cfg.querying.query_file.endswith(".json"):
                    cmap_queries = json.load(f)
                    queries = list(cmap_queries.keys())
                    self._query_cmap = {
                        k: utils.hex_to_rgb(v) for k, v in cmap_queries.items()
                    }
                else:
                    queries = [l.strip() for l in f.readlines()]
                self.add_queries(queries)

        self.messaging_service = None
        if "messaging_service" in cfg and cfg.messaging_service is not None:
            self.messaging_service = hydra.utils.instantiate(
                cfg.messaging_service,
                text_query_callback=self.add_queries if init_encoder else None,
                sync_query_callback=self.sync_query if init_encoder else None,
                debug_callback=self.get_debug_info,
            )

    @torch.inference_mode()
    def add_queries(self, queries: List[str]):
        """Adds a list of queries to query the map with at fixed intervals.

        Args:
          queries: List of string where each string is either a text query or a
            path to a local image file for image querying.
        """
        if self.encoder is None or not hasattr(self.encoder, "encode_labels"):
            raise Exception("Trying to query without a capable text encoder.")

        if isinstance(queries, str):
            queries = [queries]

        with self._query_lock:
            queries = set(queries).difference(self._queries_labels_history)

            queries = list(queries)
            if len(queries) == 0:
                return

            self._queries_labels_history.update(queries)

        logger.info("Received queries: %s", str(queries))
        img_queries = [x for x in queries if os.path.exists(x)]
        text_queries = [x for x in queries if not os.path.exists(x)]
        text_queries_feats = None
        if len(text_queries) > 0:
            if self.cfg.querying.text_query_mode == "labels":
                text_queries_feats = self.encoder.encode_labels(text_queries)
            elif self.cfg.querying.text_query_mode == "prompts":
                text_queries_feats = self.encoder.encode_prompts(text_queries)
            else:
                raise ValueError("Invalid query type")

        img_queries_feats = None
        if len(img_queries) > 0:
            imgs = list()
            for q in img_queries:
                imgs.append(
                    torch.nn.functional.interpolate(
                        torchvision.io.read_image(q).unsqueeze(0).float().cuda() / 255,
                        size=(self.dataset.rgb_h, self.dataset.rgb_w),
                        mode="bilinear",
                        antialias=True,
                    )
                )
            imgs = torch.cat(imgs, dim=0)
            img_queries_feats = self.encoder.align_global_features_with_language(
                self.encoder.encode_image_to_vector(imgs)
            )

        queries_labels = dict(text=text_queries, img=img_queries)
        queries_feats = dict(text=text_queries_feats, img=img_queries_feats)
        if self.feat_compressor is not None and self.cfg.querying.compressed:
            if not self.feat_compressor.is_fitted():
                logger.warning(
                    "The feature compressor was not fitted. "
                    "Will try to fit to query features which may fail."
                )
                l = [x for x in queries_feats.values() if x is not None]
                self.feat_compressor.fit(torch.cat(l, dim=0))
            for k, v in queries_feats.items():
                if v is None:
                    continue
                queries_feats[k] = self.feat_compressor.compress(v)

        with self._query_lock:
            if self._queries_feats is None:
                self._queries_labels = queries_labels
                self._queries_feats = queries_feats
            else:
                for k, v in queries_feats.items():
                    if v is None:
                        continue
                    if k not in self._queries_feats:
                        self._queries_feats[k] = v
                        self._queries_labels[k] = queries_labels
                    else:
                        self._queries_feats[k] = torch.concat(
                            (self._queries_feats[k], queries_feats[k]), dim=0
                        )
                        self._queries_labels[k].extend(queries_labels[k])

            self._queries_updated = True

    def clear_queries(self):
        with self._query_lock:
            self._queries_labels = None
            self._queries_feats = None
            self._queries_updated = False

    def run_queries(self):
        with self._query_lock:
            if self._queries_feats is not None and len(self._queries_feats) > 0:
                kwargs = dict()
                if self._query_cmap is not None and len(self._query_cmap) > 0:
                    kwargs["vis_colors"] = self._query_cmap

                for k, v in self._queries_labels.items():
                    if v is None or len(v) < 1:
                        continue
                    r = self.mapper.feature_query(
                        self._queries_feats[k],
                        softmax=self.cfg.querying.compute_prob,
                        compressed=self.cfg.querying.compressed,
                    )
                    if self.vis is not None and r is not None:
                        self.mapper.vis_query_result(r, vis_labels=v, **kwargs)

                self._queries_updated = False
            with self._status_lock:
                if (
                    self.status == MappingServer.Status.IDLE
                    and not self.cfg.querying.compute_prob
                ):
                    # No need to relog old queries so we clear them.
                    self._queries_feats = None
                    self._queries_labels.clear()

    def get_debug_info(self) -> dict:
        """Return calibration and pose debug info for diagnostics."""
        from rayfronts import geometry3d as g3d

        info = {}

        # Dataset calibration
        ds = self.dataset
        if hasattr(ds, "intrinsics_3x3") and ds.intrinsics_3x3 is not None:
            K = ds.intrinsics_3x3
            info["intrinsics"] = {
                "fx": float(K[0, 0]),
                "fy": float(K[1, 1]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2]),
            }
        if hasattr(ds, "_pose_to_camera_transform"):
            info["pose_to_camera_translation"] = ds._pose_to_camera_transform[
                :3, 3
            ].tolist()
        if hasattr(ds, "src2rdf_transform"):
            info["src2rdf_transform"] = ds.src2rdf_transform.tolist()

        src_coord = getattr(self.cfg.dataset, "src_coord_system", "flu")
        info["src_coord_system"] = src_coord

        # Latest robot pose
        if self._last_pose_rdf is not None:
            rdf_t = self._last_pose_rdf[:3, 3]
            rdf2src = g3d.get_coord_system_transform("rdf", src_coord)
            odom_t = (rdf2src @ rdf_t).numpy().tolist()
            info["robot_pose_rdf"] = rdf_t.tolist()
            info["robot_pose_odom"] = odom_t
        else:
            info["robot_pose_rdf"] = None
            info["robot_pose_odom"] = None

        # Map stats
        info["map_empty"] = self.mapper.is_empty()
        if (
            hasattr(self.mapper, "global_vox_xyz")
            and self.mapper.global_vox_xyz is not None
        ):
            info["num_voxels"] = int(self.mapper.global_vox_xyz.shape[0])
        if hasattr(self.cfg.mapping, "vox_size"):
            info["vox_size"] = float(self.cfg.mapping.vox_size)

        return info

    @torch.inference_mode()
    def sync_query(self, query_text: str) -> dict:
        """Process a single text query synchronously and return results.

        Args:
            query_text: The text to query for.

        Returns:
            Dict containing:
              - query: The original query text
              - found: Whether a match was found
              - score: Similarity score of best single voxel (0-1)
              - nav_goal: [x, y] absolute target in dataset source frame
              - nav_goal_rdf: [x, y, z] absolute target in world RDF frame
              - nav_goal_flu: [x, y, z] absolute target in FLU frame
              - position_description: Human-readable absolute coordinates
              - position_description_flu: Human-readable absolute FLU coordinates
              - num_matches: Number of voxels above threshold
        """
        if self.encoder is None or not hasattr(self.encoder, "encode_labels"):
            return {
                "query": query_text,
                "found": False,
                "error": "No encoder available",
            }

        if self.mapper.is_empty():
            return {"query": query_text, "found": False, "error": "Map is empty"}

        try:
            # Encode the query text (default to "prompts" if not specified)
            query_mode = (
                getattr(self.cfg.querying, "text_query_mode", None) or "prompts"
            )
            if query_mode == "labels":
                query_feat = self.encoder.encode_labels([query_text])
            else:  # "prompts" or any other value defaults to prompts
                query_feat = self.encoder.encode_prompts([query_text])

            # Compress if needed
            if (
                self.feat_compressor is not None
                and self.cfg.querying.compressed
                and self.feat_compressor.is_fitted()
            ):
                query_feat = self.feat_compressor.compress(query_feat)

            # Query the map (softmax=False for single queries - softmax over 1 query is meaningless)
            r = self.mapper.feature_query(
                query_feat, softmax=False, compressed=self.cfg.querying.compressed
            )

            if r is None:
                return {
                    "query": query_text,
                    "found": False,
                    "error": "Query returned no results",
                }

            if self.vis is not None:
                self.vis.step()
                self.mapper.vis_query_result(r, vis_labels=[query_text])

            # Extract best match
            vox_xyz = r.get("vox_xyz", r.get("pc_xyz"))
            vox_sim = r.get("vox_sim", r.get("pc_sim"))

            if vox_xyz is None or vox_sim is None:
                return {
                    "query": query_text,
                    "found": False,
                    "error": "No position data",
                }

            # Get best match (first query index since we only have one).
            # Keep ranking/counting robust to non-finite values.
            sim_scores = vox_sim[0, :] if vox_sim.dim() > 1 else vox_sim
            finite_mask = torch.isfinite(sim_scores)
            valid_scores = sim_scores[finite_mask]
            valid_xyz = vox_xyz[finite_mask]

            if valid_scores.numel() > 0:
                best_score = float(valid_scores.max().item())
            else:
                best_score = float("nan")

            above_thresh = valid_scores > DEFAULT_QUERY_THRESHOLD
            num_matches = int(above_thresh.sum().item())

            # Check if best match meets confidence threshold
            found = bool(
                valid_scores.numel() > 0 and best_score >= DEFAULT_QUERY_THRESHOLD
            )

            # ------------------------------------------------------------------
            # Nav target: top-k weighted centroid
            # Use the top scoring voxels and compute a score-weighted centroid.
            # ------------------------------------------------------------------
            best_rdf = None
            nav_goal_world = None  # [x, y] absolute target in source frame

            if found and num_matches > 0:
                top_k = min(10, valid_scores.numel())
                top_scores, top_idx = torch.topk(valid_scores, k=top_k)
                top_xyz = valid_xyz[top_idx].cpu()
                top_scores_cpu = top_scores.cpu()

                # Keep weights non-negative; fallback to uniform if all are <= 0.
                top_scores_pos = torch.clamp(top_scores_cpu, min=0.0)
                if float(top_scores_pos.sum().item()) > 0:
                    top_weights = top_scores_pos / top_scores_pos.sum()
                else:
                    top_weights = torch.ones_like(top_scores_pos) / top_k

                best_rdf = (top_weights.unsqueeze(-1) * top_xyz).sum(dim=0)  # (3,)

                logger.info(
                    f"Query '{query_text}': top-{top_k} weighted centroid "
                    f"for nav_goal_rdf={[round(v, 3) for v in best_rdf.tolist()]}"
                )

                # Convert absolute RDF target into dataset source frame and keep [x, y].
                src_coord = getattr(self.cfg.dataset, "src_coord_system", "flu")
                from rayfronts import geometry3d as g3d

                rdf2src = g3d.get_coord_system_transform("rdf", src_coord)
                best_src = (rdf2src @ best_rdf).numpy().tolist()
                nav_goal_world = [best_src[0], best_src[1]]

                logger.info(
                    f"Query '{query_text}': "
                    f"nav_goal_{src_coord}_2d={[round(v, 3) for v in nav_goal_world]}, "
                    f"nav_goal_rdf={[round(v, 3) for v in best_rdf.tolist()]}, "
                    f"topk={top_k}, "
                    f"score={best_score:.3f}, matches={num_matches}"
                )

            # Log score distribution statistics
            num_voxels = sim_scores.numel()
            score_min = float(sim_scores.min().item())
            score_max = float(sim_scores.max().item())
            score_mean = float(sim_scores.mean().item())
            score_std = float(sim_scores.std().item())
            score_median = float(sim_scores.median().item())

            # Percentiles for understanding distribution shape
            sorted_scores = sim_scores.sort().values
            p25 = float(sorted_scores[int(0.25 * num_voxels)].item())
            p75 = float(sorted_scores[int(0.75 * num_voxels)].item())
            p90 = float(sorted_scores[int(0.90 * num_voxels)].item())
            p95 = float(sorted_scores[int(0.95 * num_voxels)].item())
            p99 = float(sorted_scores[int(0.99 * num_voxels)].item())

            # Count voxels in score ranges
            below_0 = int((sim_scores < 0).sum().item())
            range_0_01 = int(((sim_scores >= 0) & (sim_scores < 0.1)).sum().item())
            range_01_02 = int(((sim_scores >= 0.1) & (sim_scores < 0.2)).sum().item())
            range_02_03 = int(((sim_scores >= 0.2) & (sim_scores < 0.3)).sum().item())
            range_03_04 = int(((sim_scores >= 0.3) & (sim_scores < 0.4)).sum().item())
            range_04_plus = int((sim_scores >= 0.4).sum().item())

            logger.info(f"Query '{query_text}': score={best_score:.3f}, found={found}")
            logger.info(f"  Score distribution ({num_voxels} voxels):")
            logger.info(
                f"    Stats: min={score_min:.4f}, max={score_max:.4f}, mean={score_mean:.4f}, std={score_std:.4f}, median={score_median:.4f}"
            )
            logger.info(
                f"    Percentiles: p25={p25:.4f}, p75={p75:.4f}, p90={p90:.4f}, p95={p95:.4f}, p99={p99:.4f}"
            )
            logger.info(
                f"    Ranges: <0={below_0}, [0,0.1)={range_0_01}, [0.1,0.2)={range_01_02}, [0.2,0.3)={range_02_03}, [0.3,0.4)={range_03_04}, >=0.4={range_04_plus}"
            )
            logger.info(
                f"    Threshold={DEFAULT_QUERY_THRESHOLD}: {num_matches} voxels ({100 * num_matches / num_voxels:.1f}%) above threshold"
            )

            # Build a simple absolute-coordinate description in RDF.
            position_description = None
            nav_goal_flu = None
            position_description_flu = None
            if found and best_rdf is not None:
                x, y, z = best_rdf.tolist()
                position_description = f"absolute RDF (x={x:.2f}, y={y:.2f}, z={z:.2f})"
                # RDF -> FLU: [x_r, y_r, z_r] -> [x_f, y_f, z_f] = [z_r, -x_r, -y_r]
                nav_goal_flu = [z, -x, -y]
                fx, fy, fz = nav_goal_flu
                position_description_flu = (
                    f"absolute FLU (x={fx:.2f}, y={fy:.2f}, z={fz:.2f})"
                )

            return {
                "query": query_text,
                "found": found,
                "score": best_score,
                "nav_goal": nav_goal_world,
                "nav_goal_rdf": best_rdf.tolist() if best_rdf is not None else None,
                "nav_goal_flu": nav_goal_flu,
                "position_description": position_description,
                "position_description_flu": position_description_flu,
                "num_matches": num_matches,
            }

        except Exception as e:
            logger.error(f"Query error: {e}")
            return {"query": query_text, "found": False, "error": str(e)}

    @torch.inference_mode()
    def run(self):
        total_wall_t0 = time.time()
        total_map = 0
        total_frames_processed = 0
        wall_t0 = time.time()

        dataloader = list()
        with self._status_lock:
            if self.status == MappingServer.Status.INIT:
                self.status = MappingServer.Status.MAPPING
                dataloader = torch.utils.data.DataLoader(
                    self.dataset, batch_size=self.cfg.batch_size
                )
                logger.info("Datastream opened. Starting mapping.")

        for i, batch in enumerate(dataloader):
            if batch is None:
                break
            rgb_img = batch["rgb_img"].cuda()
            depth_img = batch["depth_img"].cuda()
            pose_4x4 = batch["pose_4x4"].cuda()
            kwargs = dict()
            if "confidence_map" in batch.keys():
                kwargs["conf_map"] = batch["confidence_map"].cuda()

            if self.cfg.depth_limit >= 0:
                depth_img[
                    torch.logical_and(
                        torch.isfinite(depth_img), depth_img > self.cfg.depth_limit
                    )
                ] = torch.inf

            # Visualize inputs
            if self.vis is not None:
                # Set time BEFORE logging for this frame
                self.vis.step()
                if self.cfg.vis.pose_period > 0 and i % self.cfg.vis.pose_period == 0:
                    self.vis.log_pose(batch["pose_4x4"][-1])
                if self.cfg.vis.input_period > 0 and i % self.cfg.vis.input_period == 0:
                    self.vis.log_img(batch["rgb_img"][-1].permute(1, 2, 0))
                    self.vis.log_depth_img(depth_img.cpu()[-1].squeeze())
                    if "confidence_map" in batch.keys():
                        self.vis.log_img(batch["confidence_map"][-1])
                    if "semseg_img" in batch.keys():
                        self.vis.log_label_img(batch["semseg_img"][-1])

            # Keep the latest pose (RDF) so sync_query can compute "closest to robot"
            self._last_pose_rdf = pose_4x4[-1].cpu()  # (4, 4) last frame in batch

            map_t0 = time.time()
            r = self.mapper.process_posed_rgbd(rgb_img, depth_img, pose_4x4, **kwargs)
            map_t1 = time.time()

            if self.vis is not None:
                if self.cfg.vis.input_period > 0 and i % self.cfg.vis.input_period == 0:
                    self.mapper.vis_update(**r)
                if self.cfg.vis.map_period > 0 and i % self.cfg.vis.map_period == 0:
                    self.mapper.vis_map()

            if self.cfg.querying.period > 0 and i % self.cfg.querying.period == 0:
                self.run_queries()

            # Stat calculation
            total_frames_processed += rgb_img.shape[0]
            map_p = map_t1 - map_t0
            total_map += map_p
            map_thr = rgb_img.shape[0] / map_p
            wall_t1 = time.time()
            wall_p = wall_t1 - wall_t0
            wall_thr = rgb_img.shape[0] / wall_p
            wall_t0 = wall_t1
            logger.info(
                "[#%4d#] Wall (#%6.4f# ms/batch - #%6.2f# frame/s), "
                "Mapping (#%6.4f# ms/batch - #%6.2f# frame/s), "
                "Mapping/Wall (#%6.4f%%)",
                i,
                wall_p * 1e3,
                wall_thr,
                map_p * 1e3,
                map_thr,
                map_p / wall_p * 100,
            )

            with self._status_lock:
                if self.status != MappingServer.Status.MAPPING:
                    logger.info("Mapping stopped.")
                    break

        # Final stat calculation
        total_wall_t1 = time.time()
        total_wall = total_wall_t1 - total_wall_t0
        if total_map > 0 and total_wall > 0:
            logger.info(
                "Total Wall (#%6.4f# s - #%6.2f# frame/s), "
                "Total Mapping (#%6.4f# s - #%6.2f# frame/s), "
                "Mapping/Wall (#%6.4f%%)",
                total_wall,
                total_frames_processed / total_wall,
                total_map,
                total_frames_processed / total_map,
                total_map / total_wall * 100,
            )

        # Shutting down or transitioning to idling
        self._status_lock.acquire()
        if self.status == MappingServer.Status.MAPPING:
            if self.messaging_service is not None:
                self.status = MappingServer.Status.IDLE
                try:
                    self.dataset.shutdown()
                except AttributeError:
                    pass  # Its fine dataset doesn't have shutdown function
            else:
                self.shutdown()
                return

        # No new data is coming so we only need to add new queries and not
        # update old ones. Unless compute_prob is set to true b.c new queries
        # will not affect old results.
        if not self.cfg.querying.compute_prob:
            self._queries_feats = None
            self._queries_labels.clear()
        # Idling loop
        while self.status == MappingServer.Status.IDLE:
            self._status_lock.release()
            time.sleep(1)
            with self._query_lock:
                if self._queries_updated:
                    self.run_queries()
            self._status_lock.acquire()

        self.status = MappingServer.Status.CLOSED
        self._status_lock.release()
        self.shutdown()

    def shutdown(self):
        with self._status_lock:
            self.status = MappingServer.Status.CLOSING
        if self.messaging_service is not None:
            self.messaging_service.shutdown()
        if self.dataset is not None:
            try:
                self.dataset.shutdown()
            except AttributeError:
                pass
        if self.vis is not None:
            try:
                self.vis.shutdown()
            except AttributeError:
                pass
        with self._status_lock:
            self.status = MappingServer.Status.CLOSED


def signal_handler(mapping_server: MappingServer, sig, frame):
    with mapping_server._status_lock:
        if mapping_server.status == MappingServer.Status.MAPPING:
            if mapping_server.messaging_service is not None:
                logger.info(
                    "Received interrupt signal. Stopping mapping. Messaging service is "
                    "still online. Interrupt again to shutdown."
                )

                mapping_server.status = MappingServer.Status.IDLE
            else:
                logger.info("Received interrupt signal. Shutting down.")
                mapping_server.status = MappingServer.Status.CLOSING

        elif mapping_server.status == MappingServer.Status.IDLE:
            logger.info("Received interrupt signal. Shutting down.")
            mapping_server.status = MappingServer.Status.CLOSING
    try:
        mapping_server.dataset.shutdown()
    except AttributeError:
        pass  # Its fine dataset doesn't have shutdown function


@hydra.main(version_base=None, config_path="configs", config_name="default")
@torch.inference_mode()
def main(cfg=None):
    if cfg.seed >= 0:
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    try:
        server = MappingServer(cfg)
    except KeyboardInterrupt:
        logger.info("Shutdown before initializing completed.")
        return

    signal.signal(signal.SIGINT, partial(signal_handler, server))
    try:
        server.run()
    except Exception as e:
        server.shutdown()
        raise e


if __name__ == "__main__":
    # Cleanup for nanobind. See https://github.com/wjakob/nanobind/issues/19
    def cleanup():
        import typing

        for cleanup in typing._cleanups:
            cleanup()

    atexit.register(cleanup)

    main()
