This folder ('all_images') should contain all images of the dataset,
named exactly as referenced in the 'path' column of the provided CSV split files
(e.g., C_5DSR_field3cam_001.tiff).

Ground-truth illuminants are stored in the provided CSV files (Lr, Lg, Lb columns).

Set the 'root' field in the config to point to this folder. The CSV 'path' entries
are resolved relative to this 'root'.

Example:
  all_images/
    C_5DSR_field3cam_001.tiff
    C_5DSR_field1cam_002.tiff
    N_D810_lab_003.tiff
    ...

Note on Intel-TAU:
- Images are used as distributed by the Intel-TAU dataset: 
  black-level corrected, and color-checker masked by the provider, etc.