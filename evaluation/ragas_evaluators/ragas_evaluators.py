from dotenv import load_dotenv
load_dotenv()

from langchain.schema import Document
from ragas.langchain.evalchain import RagasEvaluatorChain
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# create evaluation chains
faithfulness_chain = RagasEvaluatorChain(metric=faithfulness)
answer_rel_chain = RagasEvaluatorChain(metric=answer_relevancy)
context_rel_chain = RagasEvaluatorChain(metric=context_precision)
context_recall_chain = RagasEvaluatorChain(metric=context_recall)

if __name__ == "__main__":
    # example usage

    result = {'query': 'What is the significance of the Statue of Liberty in New York City?',
    'ground_truths': ['The Statue of Liberty in New York City holds great significance as a symbol of the United States and its ideals of liberty and peace. It greeted millions of immigrants who arrived in the U.S. by ship in the late 19th and early 20th centuries, representing hope and freedom for those seeking a better life. It has since become an iconic landmark and a global symbol of cultural diversity and freedom.'],
    'result': 'The Statue of Liberty in New York City holds great significance as a symbol of the United States and its ideals of liberty and peace. It greeted millions of immigrants as they arrived in the U.S. in the late 19th and early 20th centuries, representing hope and opportunity. It has become an iconic landmark and a representation of freedom and cultural diversity.',
    'source_documents': [Document(page_content='from 1785 until 1790, and has been the largest U.S. city since 1790. The Statue of Liberty greeted millions of immigrants as they came to the U.S. by ship in the late 19th and early 20th centuries, and is a symbol of the U.S. and its ideals of liberty and peace. In the 21st century, New York City has emerged as a global node of creativity, entrepreneurship, and as a symbol of freedom and cultural diversity. The New York Times has won the most Pulitzer Prizes for journalism and remains the U.S. media\'s "newspaper of record". In 2019, New York City was voted the greatest city in the world in a survey of over 30,000 people from 48 cities worldwide, citing its cultural diversity.Many districts and monuments in New York City are major landmarks, including three of the world\'s ten most visited tourist attractions in 2013. A record 66.6 million tourists visited New York City in 2019. Times Square is the brightly illuminated hub of the Broadway Theater District, one of the world\'s busiest', metadata={'source': './nyc_wikipedia/nyc_text.txt'}),
    Document(page_content="The Statue of Liberty National Monument and Ellis Island Immigration Museum are managed by the National Park Service and are in both New York and New Jersey. They are joined in the harbor by Governors Island National Monument. Historic sites under federal management on Manhattan Island include Stonewall National Monument; Castle Clinton National Monument; Federal Hall National Memorial; Theodore Roosevelt Birthplace National Historic Site; General Grant National Memorial (Grant's Tomb); African Burial Ground National Monument; and Hamilton Grange National Memorial. Hundreds of properties are listed on the National Register of Historic Places or as a National Historic Landmark.", metadata={'source': './nyc_wikipedia/nyc_text.txt'}),
    Document(page_content="The majority of the most high-profile tourist destinations to the city are situated in Manhattan. These include Times Square; Broadway theater productions; the Empire State Building; the Statue of Liberty; Ellis Island; the United Nations headquarters; the World Trade Center (including the National September 11 Memorial & Museum and One World Trade Center); the art museums along Museum Mile; green spaces such as Central Park, Washington Square Park, the High Line, and the medieval gardens of The Cloisters; the Stonewall Inn; Rockefeller Center; ethnic enclaves including the Manhattan Chinatown, Koreatown, Curry Hill, Harlem, Spanish Harlem, Little Italy, and Little Australia; luxury shopping along Fifth and Madison Avenues; and events such as the Halloween Parade in Greenwich Village; the Brooklyn Bridge (shared with Brooklyn); the Macy's Thanksgiving Day Parade; the lighting of the Rockefeller Center Christmas Tree; the St. Patrick's Day Parade; seasonal activities such as ice", metadata={'source': './nyc_wikipedia/nyc_text.txt'}),
    Document(page_content="Tourism is a vital industry for New York City, and NYC & Company represents the city's official bureau of tourism. New York has witnessed a growing combined volume of international and domestic tourists, reflecting over 60 million visitors to the city per year, the world's busiest tourist destination. Approximately 12 million visitors to New York City have been from outside the United States, with the highest numbers from the United Kingdom, Canada, Brazil, and China. Multiple sources have called New York the most photographed city in the world.I Love New York (stylized I ❤ NY) is both a logo and a song that are the basis of an advertising campaign and have been used since 1977 to promote tourism in New York City, and later to promote New York State as well. The trademarked logo, owned by New York State Empire State Development, appears in souvenir shops and brochures throughout the city and state, some licensed, many not. The song is the state song of New York.", metadata={'source': './nyc_wikipedia/nyc_text.txt'})]}

    eval_result = faithfulness_chain(result)
    eval_result["faithfulness_score"]