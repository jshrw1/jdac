There are two .py files one which explores the data and creates the charts and maps in the first two slides of the presentation. The second creates the charts and data presented in the following slides. The presentation_charts.py files creates and saves figures in the /pngs folder. 

The model.py file downloads data on trips anf weather and uses the XGBoost model to predict the number of daily trips. 

To run the files clone the repositry and create a virtual enviromenet using the requirements.txt file to import the required dependendancies. Then run the presentation_charts.py to recreate the PNGs and then run model.py to replicate the analysis.
