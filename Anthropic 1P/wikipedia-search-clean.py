from dotenv import load_dotenv
import os
import re
import json
from openai import OpenAI
from dataclasses import dataclass
from abc import ABC, abstractmethod
import wikipedia, re
from typing import Tuple, Optional, List, Dict, Any
import tiktoken


# Load variables from .env
load_dotenv()

# Access variables
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"


# Initialize the client
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Tool Description Prompt
wikipedia_prompt = """You will be asked a question by a human user. You have access to the following tool to help answer the question. <tool_description> Search Engine Tool * The search engine will exclusively search over Wikipedia for pages similar to your query. It returns for each page its title and full page content. Use this tool if you want to get up-to-date and comprehensive information on a topic to help answer queries. Queries should be as atomic as possible -- they only need to address one part of the user's question. For example, if the user's query is "what is the color of a basketball?", your search query should be "basketball". Here's another example: if the user's question is "Who created the first neural network?", your first query should be "neural network". As you can see, these queries are quite short. Think keywords, not phrases. * At any time, you can make a call to the search engine using the following syntax: <search_query>query_word</search_query>. * You'll then get results back in <search_result> tags.</tool_description>"""

retrieval_prompt = """Before beginning to research the user's question, first think for a moment inside <scratchpad> tags about what information is necessary for a well-informed answer. If the user's question is complex, you may need to decompose the query into multiple subqueries and execute them individually. Sometimes the search engine will return empty search results, or the search results may not contain the information you need. In such cases, feel free to try again with a different query. 

After each call to the Search Engine Tool, reflect briefly inside <search_quality></search_quality> tags about whether you now have enough information to answer, or whether more information is needed. If you have all the relevant information, write it in <information></information> tags, WITHOUT actually answering the question. Otherwise, issue a new search.

Here is the user's question: <question>{query}</question> Remind yourself to make short queries in your scratchpad as you plan out your strategy."""

answer_prompt = "Here is a user query: <query>{query}</query>. Here is some relevant information: <information>{information}</information>. Please answer the question using the relevant information."


@dataclass
class SearchResult:
    """
    A single search result.
    """
    content: str

class SearchTool:
    """
    A search tool that can run a query and return a formatted string of search results.
    """

    def __init__(self):
        pass

    @abstractmethod
    def raw_search(self, query: str, n_search_results_to_use: int) -> List[SearchResult]:
        """
        Runs a query using the searcher, then returns the raw search results without formatting.

        :param query: The query to run.
        :param n_search_results_to_use: The number of results to return.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def process_raw_search_results(
        self, results: List[SearchResult],
    ) -> List[str]:
        """
        Extracts the raw search content from the search results and returns a list of strings that can be passed to the model.

        :param results: The search results to extract.
        """
        raise NotImplementedError()
    
    def search_results_to_string(self, extracted: List[str]) -> str:
        """
        Joins and formats the extracted search results as a string.

        :param extracted: The extracted search results to format.
        """
        result = "\n".join(
            [
                f'<item index="{i+1}">\n<page_content>\n{r}\n</page_content>\n</item>'
                for i, r in enumerate(extracted)
            ]
        )
        return result

    def wrap_search_results(self, extracted: List[str]) -> str:
        """
        Formats the extracted search results as a string, including the <search_results> tags.

        :param extracted: The extracted search results to format.
        """
        return f"\n<search_results>\n{self.search_results_to_string(extracted)}\n</search_results>"
    
    def search(self, query: str, n_search_results_to_use: int) -> str:
        raw_search_results = self.raw_search(query, n_search_results_to_use)
        processed_search_results = self.process_raw_search_results(raw_search_results)
        displayable_search_results = self.wrap_search_results(processed_search_results)
        return displayable_search_results 
    
@dataclass
class WikipediaSearchResult(SearchResult):
    title: str
    
class WikipediaSearchTool(SearchTool):

    def __init__(self,
                 truncate_to_n_tokens: Optional[int] = 5000):
        self.truncate_to_n_tokens = truncate_to_n_tokens
        if truncate_to_n_tokens is not None:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Using OpenAI's tokenizer

    def raw_search(self, query: str, n_search_results_to_use: int) -> List[WikipediaSearchResult]:
        search_results = self._search(query, n_search_results_to_use)
        return search_results
    
    def process_raw_search_results(self, results: List[WikipediaSearchResult]) -> List[str]:
        processed_search_results = [f'Page Title: {result.title.strip()}\nPage Content:\n{self.truncate_page_content(result.content)}' for result in results]
        return processed_search_results

    def truncate_page_content(self, page_content: str) -> str:
        if self.truncate_to_n_tokens is None:
            return page_content.strip()
        else:
            tokens = self.tokenizer.encode(page_content)
            truncated_tokens = tokens[:self.truncate_to_n_tokens]
            return self.tokenizer.decode(truncated_tokens).strip()
        
    def _search(self, query: str, n_search_results_to_use: int) -> List[WikipediaSearchResult]:
        results = wikipedia.search(query)
        search_results: List[WikipediaSearchResult] = []
        for result in results:
            if len(search_results) >= n_search_results_to_use:
                break
            try:
                page = wikipedia.page(result)
                print(page.url)
            except:
                # The Wikipedia API is a little flaky, so we just skip over pages that fail to load
                continue
            content = page.content
            title = page.title
            search_results.append(WikipediaSearchResult(content=content, title=title))
        return search_results
    
def extract_between_tags(tag: str, string: str, strip: bool = True) -> list[str]:
    ext_list = re.findall(f"<{tag}(?:\\s[^>]*)?>(.+?)</{tag}>", string, re.DOTALL)
    if strip:
        ext_list = [e.strip() for e in ext_list]
    return ext_list

class ClientWithRetrieval:

    def __init__(self, search_tool: SearchTool, client: OpenAI, verbose: bool = True):
        self.search_tool = search_tool
        self.client = client
        self.verbose = verbose
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text using tiktoken"""
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    # Helper methods
    def _search_query_stop(self, partial_completion: str, n_search_results_to_use: int) -> Tuple[List[SearchResult], str]:
        search_query = extract_between_tags('search_query', partial_completion + '</search_query>') 
        if not search_query:
            raise Exception(f'Completion with retrieval failed as partial completion returned mismatched <search_query> tags.')
        print(f'Running search query against SearchTool: {search_query}')
        search_results = self.search_tool.raw_search(search_query[0], n_search_results_to_use)
        extracted_search_results = self.search_tool.process_raw_search_results(search_results)
        formatted_search_results = self.search_tool.wrap_search_results(extracted_search_results)
        return search_results, formatted_search_results
    
    def retrieve(self,
                 query: str,
                 model: str,
                 n_search_results_to_use: int = 3,
                 max_tokens_to_sample: int = 1000,
                 max_searches_to_try: int = 5,
                 temperature: float = 1.0) -> tuple[List[SearchResult], str]:
        
        system_message = wikipedia_prompt + " " + retrieval_prompt.format(query=query)
        starting_messages = [
            {"role": "system", "content": system_message}
        ]
        messages = starting_messages.copy()
        print("Starting system message:", system_message)
        token_budget = max_tokens_to_sample
        all_raw_search_results: List[SearchResult] = []
        final_model_response = ""
        
        for tries in range(max_searches_to_try):
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=token_budget,
                temperature=temperature,
                stop=["</search_query>"]
            )
            partial_completion = response.choices[0].message.content
            print(partial_completion)
            token_budget -= self.count_tokens(partial_completion)
            final_model_response += partial_completion
            messages.append({"role": "assistant", "content": partial_completion})
            
            if "<search_query>" in partial_completion:
                print(f'Attempting search number {tries}.')
                raw_search_results, formatted_search_results = self._search_query_stop(partial_completion, n_search_results_to_use)
                messages.append({"role": "user", "content": "</search_query>" + formatted_search_results})
                all_raw_search_results += raw_search_results
            else:
                break
                
        return all_raw_search_results, final_model_response
    
    # Main methods
    def completion_with_retrieval(self,
                                  query: str,
                                  model: str,
                                  n_search_results_to_use: int = 3,
                                  max_tokens_to_sample: int = 1000,
                                  max_searches_to_try: int = 5,
                                  temperature: float = 1.0) -> str:
        
        _, retrieval_response = self.retrieve(query, model=model,
                                              n_search_results_to_use=n_search_results_to_use,
                                              max_tokens_to_sample=max_tokens_to_sample,
                                              max_searches_to_try=max_searches_to_try,
                                              temperature=temperature)
        
        information = extract_between_tags('information', retrieval_response)
        if information:
            information = information[-1]
        else:
            information = "No information found in the search results."
            
        prompt = answer_prompt.format(query=query, information=information)
        print("Summarizing:\n", prompt)
        
        answer = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1000
        ).choices[0].message.content
        
        return answer
    
# Create a searcher
wikipedia_search_tool = WikipediaSearchTool()
DEEPSEEK_MODEL = MODEL_NAME

search_client = ClientWithRetrieval(search_tool=wikipedia_search_tool, client=client, verbose=True)
information =""
query = "Does Trump have any cousins?"

augmented_response = search_client.completion_with_retrieval(
    query=query,
    model=DEEPSEEK_MODEL,
    n_search_results_to_use=1,
    max_searches_to_try=5,
    max_tokens_to_sample=1000,
    temperature=0)
print(augmented_response)