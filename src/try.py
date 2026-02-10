from pageindex import PageIndexClient
import os
from dotenv import load_dotenv

load_dotenv()   
c = PageIndexClient(api_key=os.getenv("PAGEINDEX_API_KEY"))
print(dir(c))
