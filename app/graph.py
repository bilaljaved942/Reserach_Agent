from langgraph.graph import END, START, StateGraph

from app.agents.benchmark_agent import run_benchmark_agent
from app.agents.fact_checker import run_fact_checker
from app.agents.manager import run_manager
from app.agents.paper_agent import run_paper_agent
from app.agents.search_agent import run_search_agent
from app.agents.synthesis_agent import run_synthesis_agent
from app.schemas import ResearchState


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("manager", run_manager)
    graph.add_node("search_agent", run_search_agent)
    graph.add_node("paper_agent", run_paper_agent)
    graph.add_node("benchmark_agent", run_benchmark_agent)
    graph.add_node("fact_checker", run_fact_checker)
    graph.add_node("synthesis_agent", run_synthesis_agent)

    graph.add_edge(START, "manager")

    # Manager fans out to the three specialists, which run in parallel.
    graph.add_edge("manager", "search_agent")
    graph.add_edge("manager", "paper_agent")
    graph.add_edge("manager", "benchmark_agent")

    # Fact checker waits for all three specialists to finish.
    graph.add_edge("search_agent", "fact_checker")
    graph.add_edge("paper_agent", "fact_checker")
    graph.add_edge("benchmark_agent", "fact_checker")

    graph.add_edge("fact_checker", "synthesis_agent")
    graph.add_edge("synthesis_agent", END)

    return graph.compile()


RESEARCH_GRAPH = build_graph()
