This folder ('all_images') should contain all preprocessed images of the dataset,
named exactly as referenced in the 'path' column of the provided CSV split files
(e.g., 38_8D5U5565.png).

Ground-truth illuminants are stored in the provided CSV files (Lr, Lg, Lb columns).

Set the 'root' field in the config to point to this folder. The CSV 'path' entries
are resolved relative to this 'root'.

Example:
  all_images/
    38_8D5U5565.png
    513_IMG_0839.png
    411_IMG_0676.png
    ...

Note:
- Images are used as provided.
  The dataset-provided masks applied (ColorChecker, saturated, and clipped pixels).
- See the dataset page for download and masking details.