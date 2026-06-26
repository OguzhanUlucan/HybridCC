This folder ('all_images') should contain all preprocessed images of the dataset,
named exactly as referenced in the 'path' column of the provided CSV split files
(e.g., Canon1DsMkIII_0001.tiff).

Ground-truth illuminants are stored in the provided CSV files (R, G, B columns).

Set the 'root' field in the config to point to this folder. The CSV 'path' entries
are resolved relative to this 'root'. Note that, images from all 8 cameras are placed together
in this single folder.

Example:
  all_images/
    Canon1DsMkIII_0001.tiff
    Canon600D_0001.tiff
    SonyA57_0001.tiff
    ...
