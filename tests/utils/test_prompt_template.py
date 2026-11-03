from utils.prompt_template import PromptLoader, with_vocabulary


def test_prompt_loader_returns_top_level_chat_template():
    template = PromptLoader.get_template("chat_prompt")
    assert template.startswith("You are an intelligent meeting assistant")

def test_prompt_loader_returns_top_level_agenda_prompt():
    template = PromptLoader.get_template("agenda_prompt")
    assert template.startswith("You are an agenda-tracking assistant embedded in a live meeting tool")


def test_prompt_loader_returns_task_prompt_for_language():
    prompt = PromptLoader.get_task_prompt("ASR", "en-US")
    assert prompt == "Can you transcribe this audio?"


def test_prompt_loader_normalizes_mixed_case_language_code():
    prompt = PromptLoader.get_task_prompt("ASR", "En-US")
    assert prompt == "Can you transcribe this audio?"


def test_prompt_loader_falls_back_to_english_task_prompt():
    # ASR has no "pl" entry; loader should use English fallback.
    prompt = PromptLoader.get_task_prompt("ASR", "pl-PL")
    assert prompt == "Can you transcribe this audio?"


def test_with_vocabulary_empty_keeps_prompt_unchanged():
    prompt = "Can you transcribe this audio?"
    assert with_vocabulary(prompt, "") == prompt
    assert with_vocabulary(prompt, " , ,") == prompt


def test_with_vocabulary_lists_the_names():
    prompt = with_vocabulary("Can you transcribe this audio?", "Agentar, Meetween ,")
    assert prompt == (
        "Can you transcribe this audio? "
        "The speakers may say these names: Agentar, Meetween. Spell them exactly like this."
    )
