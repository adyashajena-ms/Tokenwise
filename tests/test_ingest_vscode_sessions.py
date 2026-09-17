from app.ingest_vscode_sessions import task_type_for_workspace


def test_known_workspaces_map_to_business_task_types():
    assert task_type_for_workspace("dsatbugfixtool") == "vscode:support response"
    assert task_type_for_workspace("growth_ts_ap") == "vscode:windows app dev"
    assert task_type_for_workspace("Growth_TS_APP") == "vscode:windows app dev"
    assert task_type_for_workspace("techhelpxap") == "vscode:xap_workflows"


def test_workspace_mapping_is_case_insensitive():
    assert task_type_for_workspace("TechHelpXAP") == "vscode:xap_workflows"


def test_unknown_workspace_keeps_its_name():
    assert task_type_for_workspace("another-project") == "vscode:another-project"