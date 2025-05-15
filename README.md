MSc_Bieri
Inputs from the profesor

- **Analysing the Role of Media Sentiment in the Downfall of Credit Suisse (Master’s Thesis by N. Schubiger):** 
https://github.com/nschub/Sentiment-and-Financial-Analysis/tree/main
In this repo, you might be particularly interested in the two files “test_sentiment_analysis.py” and “main.py”. The first file includes the definition of a function to create a news sentiment from an arbitrary text provided. The second file contains a function “process_dataframe” which applies this function in practice (i.e. specifying the GPT system role, the model used, the sentiment scores (1,0,-1) assigned per news article etc.).
- **AI and Impact Investing (Prototype by HSLU):** 
https://github.com/HSLU-IFZ-Competence-Center-Investments/AI_and_Impact_Investing
In this repo, you might be particularly interested in the file “chat.py” from the CODE folder. In this file, you can find the definition of a function to analyse (synthetic) data from a company website to create a ranking of the company’s exposure to the 17 Sustainable Development Goals.



Tambien seria interesantate incluir lacuales son las noticias mas relvantesa en cuanto a bearich o bullish, por ejemplo Q earings



HTML

python -m venv temp_convert_env
.\temp_convert_env\Scripts\Activate.ps1
pip install notebook

LUEGO
python -m jupyter nbconvert --to html "C:\Users\Victor\Downloads\MSc DataScience\MasterThesis - GitHub\2. Exploratory Data Analysis.ipynb"

Y FINALMENTE

deactivate
Remove-Item -Recurse -Force temp_convert_env
