# References

## Dataset

- Das, A., Kerr, D., Glantz, N., Bevier, W., Santiago, R., Gutierrez-Osuna, R.,
  & Mortazavi, B. J. (2025). CGMacros: a pilot scientific dataset for
  personalized nutrition and diet monitoring. *Scientific Data*, 12(1), 1557.
  https://doi.org/10.1038/s41597-025-05851-7
- Gutierrez-Osuna, R., Kerr, D., Mortazavi, B., & Das, A. (2025). CGMacros
  (version 1.0.0). *PhysioNet*. https://doi.org/10.13026/3z8q-x658
- Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet:
  components of a new research resource for complex physiologic signals.
  *Circulation*, 101(23), e215-e220. (PhysioNet asks that this accompany any
  use of its data.)

Raw data can be downloaded from the PhysioNet record
above.

## Methods

- Parkes, J. L., Slatin, S. L., Pardo, S., & Ginsberg, B. H. (2000). A new
  consensus error grid to evaluate the clinical significance of inaccuracies
  in the measurement of blood glucose. *Diabetes Care*, 23(8), 1143-1148.
  (the error grid behind the clinical-safety numbers)
- Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: a gentle
  introduction. *Foundations and Trends in Machine Learning*, 16(4). (the
  uncertainty intervals)

## Background

- MetaboNet-Bench: a multi-modal benchmark for glucose forecasting in type 1
  diabetes. arXiv:2606.18640. (notes that many published forecasting
  algorithms are limited to CGM data alone)
- "Driver-Blindness". arXiv:2511.20601. (multivariate models gain almost
  nothing from meal and insulin inputs)
- Hameed, H., & Kleinberg, S. (2020). Comparing machine learning techniques
  for blood glucose forecasting using free-living and patient generated data.
  *Proceedings of Machine Learning Research*, 126 (Machine Learning for
  Healthcare). (adding meal and activity features gave the same or slightly
  worse error)
- Nemat, H., Khadem, H., Elliott, J., & Benaissa, M. (2024). Data-driven blood
  glucose level prediction in type 1 diabetes: a comprehensive comparative
  analysis. *Scientific Reports*, 14. (adding extra variables did not
  significantly improve prediction)
- Xie, J., & Wang, Q. (2020). Benchmarking machine learning algorithms on
  blood glucose prediction for type 1 diabetes in comparison with classical
  time-series models. *IEEE Transactions on Biomedical Engineering*, 67(11).
  (a classical autoregressive model on glucose history held its own against
  ten machine learning methods)
