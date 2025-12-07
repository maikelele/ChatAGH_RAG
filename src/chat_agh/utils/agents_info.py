from dataclasses import dataclass


@dataclass
class AgentDetails:
    name: str
    description: str
    cached_history: dict[str, str] | None


class AgentsInfo:
    def __init__(self, agents_details: list[AgentDetails]):
        self.agents_details = agents_details

    def __str__(self) -> str:
        text = ""
        for detail in self.agents_details:
            text += (
                f"  Agent name: {detail.name}\n"
                f"  Agent description: {detail.description}\n"
            )
            if detail.cached_history is not None:
                cached_history_text = (
                    f"\n    Query: {detail.cached_history['query']}"
                    f"\n    Response: {detail.cached_history['response']}"
                )
                text += f"  Cached Conversation: {cached_history_text}"

        return text


@dataclass
class RetrievalAgentInfo:
    name: str
    vector_store_index_name: str
    description: str


RETRIEVAL_AGENTS = [
    RetrievalAgentInfo(
        name="MAIN AGENT",
        vector_store_index_name="cluster_0",
        description="""
        Informacje ogólne i obsługa dydaktyki – Akademia Górniczo‑Hutnicza im. Stanisława Staszica w Krakowie

        Zakres treści:
	    -	Strona główna uczelni, ogólne informacje o uczelni – misja, historia, wydziały, jednostki.  ￼
	    -	Informacje dla kandydatów, studentów i doktorantów – oferta studiów, kierunki, procedury, zasoby dydaktyczne.  ￼
	    -	Jednostka zajmująca się obsługą kształcenia – Centrum Obsługi Kształcenia (COK) – regulaminy, systemy, wsparcie studenta.  ￼
	    -	Zewnętrzne i wewnętrzne serwisy uczelni: platformy, e-usługi, ogłoszenia, dane kontaktowe.  ￼
	    -	Informacje administracyjne uczelni – struktura, jednostki, dane o przetwarzaniu danych.

        Obsługuje domeny:
        - agh.edu.pl
        - cok.agh.edu.pl
        """,
    ),
    RetrievalAgentInfo(
        name="SKOS AGENT",
        vector_store_index_name="cluster_6",
        description="""
        System informacji o pracownikach & jednostkach – SkOs AGH

        Zakres treści (scope):
	    -	Baza informacji kontaktowych pracowników, ich afiliacji, jednostek, stanowisk w AGH.  ￼
	    -	Informacje o strukturze organizacyjnej uczelni: jednostki, wydziały, instytuty, ciała kolegialne, etc.  ￼
	    -	Funkcje wyszukiwania (podstawowe, zaawansowane) w systemie SkOs, np. wyszukiwanie pracownika po nazwisku, jednostce.  ￼

	    Obsługuje domeny:
	    - skos.agh.edu.pl
        - old.skos.agh.edu.pl
        """,
    ),
    RetrievalAgentInfo(
        name="SPRAWY STUDENCKIE AGENT",
        vector_store_index_name="cluster_7",
        description="""
        Zakwaterowanie & sprawy studenckie – kampus i wsparcie studenta Akademia Górniczo‑Hutnicza im. Stanisława Staszica (AGH)”

        Zakres treści (scope):
	    -	Kompleks akademików i osiedla studenckiego: Miasteczko Studenckie AGH – informacje o domach studenckich, zakwaterowaniu, cennikach, standardach pokoi.  ￼
	    -	Procesy i regulaminy przydziału miejsc w domach studenckich: wnioski, harmonogramy, rejestracja.  ￼
	    -	Obsługa spraw studenckich: prawa i obowiązki studentów, świadczenia, pomoc, regulacje uczelni.  ￼

	    Obsługuje domeny:
	    - miasteczko.agh.edu.pl
        - akademik.agh.edu.pl
        - rownosc.agh.edu.pl
        - dss.agh.edu.pl
        """,
    ),
    RetrievalAgentInfo(
        name="REKRUTACJA AGENT",
        vector_store_index_name="cluster_8",
        description="""
        Nazwa klastra: „Rekrutacja, szkolenia i promocja – usługi kształcenia i wizerunku Akademia Górniczo‑Hutnicza im Stanisława Staszica (AGH)”

        Zakres treści (scope):
        - Proces rekrutacji kandydatów na studia – portal rekrutacja.agh.edu.pl: terminy, warunki, procedury, skierowany do kandydatów.
        - Oferta kursów i szkoleń (zarówno wewnętrznych dla pracowników/studentów, jak i zewnętrznych) – serwis „Szkolenia” AGH.  ￼
        - Program strategiczny „Inicjatywa Doskonałości Uczelnia Badawcza (IDUB)” – działania badawcze, inwestycje AGH w badania, rozwój kadry i infrastruktury.  ￼￼
        - Strona marki, wizerunku i komunikacji uczelni – Centrum Komunikacji i Marketingu AGH (CKiM) odpowiadająca za portal uczelni i działania promocyjne.  ￼

        Obsługuje domeny:
        - rekrutacja.agh.edu.pl
        - ckim.agh.edu.pl
        - szkolenia.agh.edu.pl
        - idub.agh.edu.pl
        """,
    ),
    RetrievalAgentInfo(
        name="SYLABUSY AGENT",
        vector_store_index_name="cluster_9",
        description="""
        Nazwa klastra: „Zamówienia publiczne & sylabusy-oferta dydaktyczna – Akademia Górniczo‑Hutnicza im. Stanisława Staszica (AGH)”

        Zakres treści (scope):
        - Platforma Sylabusy AGH (sylabusy.agh.edu.pl) – katalog kierunków, programy studiów, moduły, efekty kształcenia, sylabusy przedmiotów – zarówno w języku polskim jak i angielskim.  ￼
        - Serwis Dział Zamówień Publicznych AGH (dzp.agh.edu.pl) – ogłoszenia przetargowe, plany zamówień uczelni, informacje formalne związane z zamówieniami publicznymi w AGH.  ￼
        - Zdaniem tej domeny: dokumenty dydaktyczne (opis przedmiotów, programy, sylabusy) oraz dokumentacja formalna uczelni związana z zamówieniami (dla AGH jako instytucji publicznej)

        Obsługuje domeny:
        - dzp.agh.edu.pl
        - sylabusy.agh.edu.pl
        """,
    ),
    # RetrievalAgentInfo(
    #     name="dss_agent",
    #     vector_store_index_name="dss",
    #     description="""
    #     Agent wyszukujący informacji z działu spraw studenckich AGH.
    #     Informacje o procedurach administracyjnych, takich jak składanie wniosków o stypendia,
    #      urlopy dziekańskie czy wydawanie zaświadczeń.
    #       Dodatkowo często dostępne są wzory dokumentów, regulaminy studiów i informacje dotyczące akademików, pomocy materialnej czy szeroko pojętej działąlności uczelni.
    #     """,
    # ),
]
