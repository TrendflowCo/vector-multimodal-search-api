from googletrans import Translator
import re

def translate(text, source_language):
    translator = Translator()
    translation = translator.translate(text, src=source_language, dest='en')
    return translation.text

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def extract_price_info(query):
    # Define a regular expression pattern to match prices
    price_pattern = r'(\d+(?:\.\d{1,2})?)\s*(?:dollars|dollar|euros|euro|\$|€)'
    
    # Search for the price pattern in the query
    price_match = re.search(price_pattern, query, re.IGNORECASE)
    
    if price_match:
        # Extract the matched price as a float
        price = float(price_match.group(1))
        
        # Extract the currency
        currency = "dollars" if "dollar" in price_match.group(0).lower() else "euros"
        
        # Remove the price part from the query
        query_without_price = re.sub(price_pattern, '', query, re.IGNORECASE).strip()
    else:
        # If no price is found, set price and currency to None
        price = None
        currency = None
        query_without_price = query

    return int(price) if price is not None else None, currency, query_without_price


def extract_negative_keywords(text):
    """
    Extracts words following each "-" in the given text.

    Parameters:
    text (str): The input text from which to extract words.

    Returns:
    list: A list of words or phrases following each "-" character.
    """
    # Regular expression pattern to find words following "-"
    pattern = r'-(\w+|\w+(?: \w+)*)'
    
    # Find all occurrences of the pattern
    matches = re.findall(pattern, text)

    return matches

