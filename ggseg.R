library(ggsegExtra)
atlas3d <- make_volumetric_ggseg3d(
  label_file = "atlas_neuromorphometrics.nii.gz",
  label_lut  = "labels_neuromorphometrics.csv",
  output_dir = "./miccai_build"
)
atlas2d <- make_ggseg3d_2_ggseg(atlas3d, steps = 1:7)
saveRDS(atlas2d, "miccai_atlas2d.rds")