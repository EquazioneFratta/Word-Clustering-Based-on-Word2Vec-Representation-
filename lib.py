import requests

def download_gutenberg_book(url_or_id):
    #TODO: estrare l'ID del libro da un URL e poi sotto ricrea l'URL con l'ide estratto, mi sembra una cosa idiota
    # se l'input è un URL, estrai l'ID del libro
    if isinstance(url_or_id, str) and url_or_id.startswith('http'):
        match = re.search(r'/(\d+)/?$', url_or_id)
        if match:
            book_id = match.group(1)
        else:
            raise ValueError("Impossibile estrarre l'ID del libro dall'URL")
    else:
        book_id = str(url_or_id)
    
    # Construct the download URL for the plain text version
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.HTTPError: #TODO: sta cosa dell'url alternativo non mi convince
        # url alternativo
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        response = requests.get(url)
        response.raise_for_status()
        return response.text