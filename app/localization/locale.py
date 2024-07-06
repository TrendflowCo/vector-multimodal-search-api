# Locale-specific configurations and constants

# Date formats by locale
date_formats = {
    'en_US': '%m/%d/%Y',  # U.S. style
    'en_GB': '%d/%m/%Y',  # British style
    'de_DE': '%d.%m.%Y',  # German style
}

# Currency symbols by locale
currency_symbols = {
    'USD': '$',  # US Dollar
    'EUR': '€',  # Euro
    'GBP': '£',  # British Pound
}

# Numeric formats by locale (example: decimal and thousands separators)
numeric_formats = {
    'en_US': {'decimal': '.', 'thousands': ','},
    'en_GB': {'decimal': '.', 'thousands': ','},
    'de_DE': {'decimal': ',', 'thousands': '.'},
}

# Specific text that varies by locale
welcome_messages = {
    'en_US': "Welcome!",
    'en_GB': "Welcome!",
    'de_DE': "Willkommen!",
    'es_ES': "¡Bienvenido!",
}

# More locale-specific data can be added as needed
