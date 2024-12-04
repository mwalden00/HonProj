# Honours Project: Copula-GPFA framework

This is the main directory for the Copula-GPFA framework for neuronal spike data analysis. 
If something doesn't work, please email me at M.Walden@sms.ed.ac.uk!

Culminated in submition to TMLR: [OpenReview](https://openreview.net/forum?id=hQIs1CHF6H)



# Installing Copula-GP from Github repo

In a virtual environment (e.g. [virtualenv](https://pypi.org/project/virtualenv/)), install all the dependencies and the package using the following commands:
```
pip install -r requirements.txt
pip install .
```

This will install the local implementation of Copula-GP with bagging and bug fixes, found in `src/copulagp`.

# Reproducability

For bagging validation test reproducibiltiy, use the `scripts/validate_bagging.py` script. For pupil dilation copula extraction, first download the data via `scripts/ecephys-download.pu`. Then, apply GPFA via `scripts/trajectory-extraction.py`. For unbagged pupil dilation entropy extraction, first run the `scripts/train_copulas.py` file and extract entropies with the `scripts/extract_entropies.py` file. For bagged pupil dilation entropy extraction, use the `scripts/train_segments_pupil.py` file, which trains, aggregates, and extracts entropies all in one.

# Citation of Copula-GP

If you find the Copula-GP package useful, please consider citing the original work:

```
@article{kudryashova2022parametric,
  title={Parametric Copula-GP model for analyzing multidimensional neuronal and behavioral relationships},
  author={Kudryashova, Nina and Amvrosiadis, Theoklitos and Dupuy, Nathalie and Rochefort, Nathalie and Onken, Arno},
  journal={PLoS computational biology},
  volume={18},
  number={1},
  pages={e1009799},
  year={2022},
  publisher={Public Library of Science San Francisco, CA USA}
}
```
