from rpg_gm.game.state import GameState


def test_create_new_state(tmp_path):
    db = tmp_path / "game.db"
    state = GameState(str(db))
    try:
        state.init("testworld", "village_square")
        assert state.get_location() == "village_square"
        assert "village_square" in state.get_visited()
    finally:
        state.close()


def test_move_location(tmp_path):
    db = tmp_path / "game.db"
    state = GameState(str(db))
    try:
        state.init("testworld", "village_square")
        state.set_location("dark_forest")
        assert state.get_location() == "dark_forest"
        visited = state.get_visited()
        assert "village_square" in visited
        assert "dark_forest" in visited
    finally:
        state.close()


def test_journal(tmp_path):
    db = tmp_path / "game.db"
    state = GameState(str(db))
    try:
        state.add_journal_entry("Arrived at the village.")
        state.add_journal_entry("Met the blacksmith.")
        journal = state.get_journal()
        assert len(journal) == 2
        assert journal[0]["text"] == "Arrived at the village."
        assert journal[1]["text"] == "Met the blacksmith."
        assert "time" in journal[0]
    finally:
        state.close()


def test_discovered_entities(tmp_path):
    db = tmp_path / "game.db"
    state = GameState(str(db))
    try:
        state.discover("npcs", "blacksmith")
        state.discover("locations", "dark_forest")
        state.discover("npcs", "blacksmith")  # duplicate, should be ignored
        discovered = state.get_discovered()
        assert "blacksmith" in discovered["npcs"]
        assert "dark_forest" in discovered["locations"]
        assert len(discovered["npcs"]) == 1
    finally:
        state.close()


def test_conversation_history(tmp_path):
    db = tmp_path / "game.db"
    state = GameState(str(db))
    try:
        state.add_message("user", "Hello")
        state.add_message("assistant", "Welcome, adventurer.")
        messages = state.get_recent_messages()
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Welcome, adventurer."}
    finally:
        state.close()


def test_conversation_history_limit(tmp_path):
    db = tmp_path / "game.db"
    state = GameState(str(db))
    try:
        for i in range(10):
            state.add_message("user", f"msg-{i}")
        messages = state.get_recent_messages(limit=3)
        assert len(messages) == 3
        assert messages[0]["content"] == "msg-7"
        assert messages[1]["content"] == "msg-8"
        assert messages[2]["content"] == "msg-9"
    finally:
        state.close()
