# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2025-01-XX

### Added - Inspired by Arize Phoenix & DSPy/GEPA

#### New Optimization Strategies
- **MetaPromptOptimizer**: LLM-based reflection for prompt improvement (Arize-inspired)
- **ReflectionOptimizer**: Reflection-based evolution with analysis of what works (GEPA-inspired)
- **FewShotOptimizer**: Optimizes both template and few-shot examples
- **BootstrapOptimizer**: Automatically generates examples from successful runs (DSPy-inspired)

#### Core Features
- **Pareto Frontier Tracking**: Maintains multiple complementary strategies instead of single best (GEPA-inspired)
- **Prompt Signatures**: Declarative input/output specifications (DSPy-style)
- **Prompt Version Management**: Track and compare prompt versions over time
- **Enhanced Evaluation**: Multi-objective optimization with weighted metrics

#### Documentation
- Comprehensive comparison with Arize Phoenix and DSPy/GEPA
- Advanced examples for all new strategies
- Getting started guide updated with new features

### Changed
- Bumped version to 0.2.0
- Enhanced core optimizer with reflection capabilities
- Improved evaluation system for multi-objective optimization

## [0.1.0] - 2025-01-XX

### Added - Initial Release

#### Core Framework
- Base `Optimizer`, `Evaluator`, and `Prompt` abstractions
- `OptimizationResult` with history tracking
- `PromptTemplate` for structured prompt generation

#### Optimization Strategies
- **GeneticOptimizer**: Genetic algorithm with crossover and mutation
- **HillClimbingOptimizer**: Local search optimization
- **SimulatedAnnealingOptimizer**: Temperature-based stochastic search
- **RandomSearchOptimizer**: Random sampling baseline

#### LLM Providers
- **AnthropicProvider**: Support for Claude models
- **OpenAIProvider**: Support for GPT models
- Base `LLMProvider` interface for extensibility

#### Features
- **Metrics**: Common evaluation metrics (accuracy, similarity, efficiency)
- **Experiment Tracking**: Save/load/compare optimization runs
- **Configuration**: YAML/JSON config support with Pydantic validation
- **Multi-objective Evaluation**: Weighted metric optimization

#### Examples & Documentation
- Basic optimization examples
- Multi-objective optimization example
- Configuration-based example
- API documentation
- Getting started guide

#### Development
- pytest test suite
- Type hints throughout
- Black and ruff formatting
- Modern Python packaging with pyproject.toml
