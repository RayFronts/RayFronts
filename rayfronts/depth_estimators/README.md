## Depth Estimators

This directory includes all depth estimators consuming RGB images (and
optionally auxiliary inputs such as raw depth and camera pose) and producing
metric depth estimates and related geometric quantities.

All estimators follow a template from the `base.py` file which provides
documented abstract classes.

### Available Options

- **LingBot-Depth**: Wrapper around the
  `LingBot-Depth` masked depth modeling approach
  ([GitHub](https://github.com/robbyant/lingbot-depth)).

### Adding a Depth Estimator

0. Read the top-level `CONTRIBUTING.md` file.
1. Create a new Python file with the same name as your estimator.
2. Extend the `DepthEstimator` base abstract class found in `base.py`.
3. Implement and override the inherited methods.
4. (Optional) Add a config file with your constructor arguments under
   `configs/depth_estimator`.
5. Import your estimator in the `depth_estimators/__init__.py` file.
6. Edit this README to include your new addition.
7. If your estimator relies on multiple supporting files, then:

   - **Option 1**: Add it as a submodule and keep a single stub file in this
     repo which contains the estimator class defined by `DepthEstimator`.
   - **Option 2**: If few supporting files are needed, add them as a directory
     which has the same name `X` as the estimator. Inside the directory there
     should be an `X.py` which defines your estimator class and an
     `__init__.py` which imports your estimator class such that it is available
     from the `X` module (not `X.X`).

