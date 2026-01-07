from langchain_core.messages import HumanMessage

from chat_agh.graph import ChatGraph
from chat_agh.utils.chat_history import ChatHistory
from chat_agh.utils.singletons import logger

if __name__ == "__main__":
    chat_graph = ChatGraph()

    chat_history = ChatHistory(messages=[HumanMessage("Hejka, kto jest rektorem AGH?")])
    logger.info("START")
    for c in chat_graph.stream(chat_history):
        print(c)
