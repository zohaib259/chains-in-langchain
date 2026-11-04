from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults

search_run = DuckDuckGoSearchRun()

response = search_run.invoke("Latest developments in AI")
print("Search Run Output:")
print(response)

search_results = DuckDuckGoSearchResults(max_results=3)

detailed_response = search_results.invoke("LangChain latest features")
print("\nDetailed Search Results:")
print(detailed_response)