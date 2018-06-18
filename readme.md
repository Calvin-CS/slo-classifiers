# Classifiers

These are the SLO classifiers each with the supporting CSIRO-internal datasets.

Per an email from Brian Jin, Feb 6, the data used by the classifiers is
collected by a data publisher that reads tweets from ESA (Emergency
Situation Awareness) with the queries :
- *adani* : stopadani, adani, goadani
- *bhp* : bhpbilliton, bhp
- *cuesta* : cuestacoal, "cuesta coal", cuesta,cqc
- *fortescue* : fortescuenews, "fortescue metals", fortescue
- *riotinto* : riotinto, "rio tinto"
- *newmontmining* : newmont, "newmont mining"
- *santos* : santosltd, santos
- *oilsearch* : oilsearchltd, "oil search"
- *woodside* : woodside,woodsideenergy,"woodside petroleum","woodside energy"
- *ilukaresources* : iluka,ilukaresources, "iluka resources"
- *whitehavencoal* : whitehaven,whitehavencoal,"whitehaven coal"

If tweet text includes the given queries, the data publisher pushes the
tweet into the data pipeline for the further processing.
