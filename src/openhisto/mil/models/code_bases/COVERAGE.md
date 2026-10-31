# CPU coverage

Verified 34/34 real profiles in 50 tests on Python 3.12.6, PyTorch 2.6.0.

Synthetic CPU forward, finite backward and strict state-dictionary reload were tested for every profile. No datasets or pretrained weights were downloaded. GPU parity and published benchmark metrics were not tested.

| Profile | Native input | CPU forward/backward/state | Source commit |
|---|---|---|---|
| `abmil` | feature_bag | Passed | eb0434ba2795 |
| `gated_abmil` | feature_bag | Passed | eb0434ba2795 |
| `dsmil` | feature_bag | Passed | 80465ed534ac |
| `dtfd_mil` | feature_bag | Passed | 10964f4dcc27 |
| `acmil` | feature_bag | Passed | e53d19c81f38 |
| `asmil` | feature_bag | Passed | 5d062271e162 |
| `clam_sb` | feature_bag | Passed | 53e2409d4a81 |
| `clam_mb` | feature_bag | Passed | 53e2409d4a81 |
| `varpool` | feature_bag | Passed | 2e40ab4d9a12 |
| `panther` | feature_bag | Passed | e7ffae402e14 |
| `wikg` | feature_bag | Passed | eb3144fb8267 |
| `tad_graph` | prepared_graph | Passed | 696f2196adc3 |
| `resgat` | prepared_graph | Passed | 65e2833c6207 |
| `patch_gcn` | prepared_graph | Passed | 823addaee5b8 |
| `slidegraph_plus` | prepared_graph | Passed | 23fbf58371a8 |
| `cs_mil` | prepared_hierarchy | Passed | d51b97f411ca |
| `zoommil` | prepared_hierarchy | Passed | da7bb7fa7162 |
| `hag_mil` | prepared_hierarchy | Passed | 9fec1ca43903 |
| `gfr_mil` | prepared_hierarchy | Passed | 25c9b1e24a85 |
| `h2_mil` | prepared_graph | Passed | ea472fefd5ea |
| `hipt` | prepared_hierarchy | Passed | 780fafaed2e5 |
| `transmil` | feature_bag | Passed | 9d6aee57c7c7 |
| `ilra` | feature_bag | Passed | bb45b9ba9d4a |
| `rrt_mil` | feature_bag | Passed | 3320b6616b2c |
| `dgr_mil` | feature_bag | Passed | ac6257ffa768 |
| `caprmil` | feature_bag | Passed | b1f21a25e21d |
| `lin_mil` | feature_bag | Passed | 82a00672cfeb |
| `mamba_mil` | feature_bag | Passed | 9aea39efc13f |
| `nnmil` | feature_bag | Passed | e92e8a0cf72d |
| `mhim_v1` | feature_bag | Passed | 5377eb6014bb |
| `mhim_v2` | feature_bag | Passed | 9d0c91abeb1b |
| `mean_mil` | feature_bag | Passed | Original baseline |
| `max_mil` | feature_bag | Passed | Original baseline |
| `mean_std_mil` | feature_bag | Passed | Original baseline |
