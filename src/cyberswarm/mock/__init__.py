"""Lead-owned MOCK simulator (stage S3, DESIGN section 5).

**This package is a MOCK, not the real simulator.** It is the project lead's
deterministic, clearly-labelled stand-in for Raksha's future network simulator,
built so the RL environment, coordinator, uncertainty wrapper and smoke path can
run before the real simulator exists. It is never a research result: every run
it produces is labelled a mock run.

:class:`~cyberswarm.mock.simulator.MockSimulator` implements *only* the public
:class:`~cyberswarm.interfaces.Simulator` and
:class:`~cyberswarm.interfaces.GroundTruthAccessor` protocols
(``kind = "mock"``, ``simulator_kind = SimulatorKind.mock``). The package
imports only :mod:`cyberswarm.contracts`, :mod:`cyberswarm.interfaces`,
:mod:`cyberswarm.common`, NumPy and NetworkX; it imports **no**
``cyberswarm.simulator`` and no other simulator's internals, and nothing under
``src/`` outside ``runner.factory`` imports it.

Every "attack" and "defense" is an abstract state transition. Nothing in this
package shells out, opens a socket, reaches the network, or mutates a file as an
action effect.
"""

from __future__ import annotations

from .simulator import MockSimulator

__all__ = ["MockSimulator"]
