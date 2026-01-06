"""Tests for configuration parsing."""

import tempfile
from pathlib import Path

import yaml

from server.vllm.launch import VLLMConfig


class TestVLLMConfigParsing:
    """Tests for VLLMConfig YAML parsing."""

    def test_parse_minimal_config(self) -> None:
        """Test parsing a minimal configuration."""
        config_data = {
            "config_id": "test",
            "config_name": "Test Config",
            "model": "meta-llama/Llama-2-7b-hf",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            config = VLLMConfig.from_yaml(config_path)
            assert config.config_id == "test"
            assert config.config_name == "Test Config"
            assert config.model == "meta-llama/Llama-2-7b-hf"
            # Check defaults
            assert config.dtype == "float16"
            assert config.max_model_len == 4096
            assert config.gpu_memory_utilization == 0.90
            assert config.tensor_parallel_size == 1
            assert config.enforce_eager is False
        finally:
            config_path.unlink()

    def test_parse_full_config(self) -> None:
        """Test parsing a full configuration with all fields."""
        config_data = {
            "config_id": "C2",
            "config_name": "Enforce Eager",
            "model": "meta-llama/Llama-2-7b-hf",
            "dtype": "bfloat16",
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.85,
            "tensor_parallel_size": 2,
            "host": "127.0.0.1",
            "port": 9000,
            "enforce_eager": True,
            "disable_log_requests": True,
            "description": "Test description",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            config = VLLMConfig.from_yaml(config_path)
            assert config.config_id == "C2"
            assert config.dtype == "bfloat16"
            assert config.max_model_len == 2048
            assert config.gpu_memory_utilization == 0.85
            assert config.tensor_parallel_size == 2
            assert config.host == "127.0.0.1"
            assert config.port == 9000
            assert config.enforce_eager is True
            assert config.disable_log_requests is True
        finally:
            config_path.unlink()

    def test_config_hash_deterministic(self) -> None:
        """Test that config hash is deterministic."""
        config_data = {
            "config_id": "test",
            "config_name": "Test Config",
            "model": "meta-llama/Llama-2-7b-hf",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            config1 = VLLMConfig.from_yaml(config_path)
            config2 = VLLMConfig.from_yaml(config_path)
            assert config1.config_hash() == config2.config_hash()
        finally:
            config_path.unlink()

    def test_to_command_base(self) -> None:
        """Test command generation for base config."""
        config = VLLMConfig(
            config_id="C1",
            config_name="Base",
            model="meta-llama/Llama-2-7b-hf",
        )

        cmd = config.to_command()

        assert "python" in cmd
        assert "-m" in cmd
        assert "vllm.entrypoints.openai.api_server" in cmd
        assert "--model" in cmd
        assert "meta-llama/Llama-2-7b-hf" in cmd
        assert "--dtype" in cmd
        assert "float16" in cmd
        assert "--enforce-eager" not in cmd

    def test_to_command_enforce_eager(self) -> None:
        """Test command generation with enforce_eager."""
        config = VLLMConfig(
            config_id="C2",
            config_name="Eager",
            model="meta-llama/Llama-2-7b-hf",
            enforce_eager=True,
        )

        cmd = config.to_command()
        assert "--enforce-eager" in cmd

    def test_parse_c1_config_file(self) -> None:
        """Test parsing the actual C1 config file."""
        config_path = Path("server/vllm/configs/C1_vllm_fp16_base.yaml")
        if config_path.exists():
            config = VLLMConfig.from_yaml(config_path)
            assert config.config_id == "C1"
            assert config.enforce_eager is False

    def test_parse_c2_config_file(self) -> None:
        """Test parsing the actual C2 config file."""
        config_path = Path("server/vllm/configs/C2_vllm_enforce_eager.yaml")
        if config_path.exists():
            config = VLLMConfig.from_yaml(config_path)
            assert config.config_id == "C2"
            assert config.enforce_eager is True
