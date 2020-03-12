# Generalized Dynamic Factor Models

Matlab code to estimate the Generalized Dynamic Factor Model as described in [Hallin, Lippi, and Reichlin (2000)](https://www.jstor.org/stable/2646650?seq=1#metadata_info_tab_contents). Code is taken from Matteo Barigozzi's [website](http://www.barigozzi.eu/Home.html); none of the code is mine.


## Functions
* `gdfm_twosided`: estimates the Generalized Dynamic Factor Model as in [Hallin, Lippi, and Reichlin (2000)](https://www.jstor.org/stable/2646650?seq=1#metadata_info_tab_contents).
* `gdfm_onesided`: estimates and forecasts the Generalized Dynamic Factor Model in [Forni, Hallin, Lippi, and Reichlin (2005)](https://amstat.tandfonline.com/doi/abs/10.1198/016214504000002050).
* `gdfm_unrestricted`: estimates and forecasts the Generalized Dynamic Factor Model as in [Forni, Hallin, Lippi, and Zaffaroni (2017)](http://www.eief.it/files/2017/05/mlippi_e_altri_jofe_2017.pdf).
* `numfactors`: estimates the number of factors as in [Hallin and Liška (2007)](https://www.tandfonline.com/doi/abs/10.1198/016214506000001275).
* `spectral.m`: computes the spectral density decomposition used by all other functions.
* `exampleAR` and `exampleMA`: examples with AR or MA factor loadings.
