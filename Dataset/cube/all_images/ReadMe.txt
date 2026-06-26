This folder ('all_images') should contain all preprocessed images of the dataset,
named exactly as referenced in the 'path' column of the provided CSV split files
(e.g., cube_plus_0001.tiff).

Ground-truth illuminants are stored in the provided CSV files (R, G, B columns).

Set the 'root' field in the config to point to this folder. The CSV 'path' entries
are resolved relative to this 'root'.

Example:
  all_images/
    cube_plus_0001.tiff
    cube_plus_0002.tiff
    ...
