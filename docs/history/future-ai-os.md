> **Historical:** This document predates the V2 redesign.

# Future Direction: the AI-Native OS

The Minecraft MVP is the *first Program*, not the whole project. The
long-term goal is an AI-native operating system designed around the same
Program interface the runtime uses today.

## From apps to Programs

In a conventional OS, apps are isolated visual tools designed for human
eyes and hands. In an AI-native OS, apps become **Programs** that expose:

- observations
- actions
- goals
- state transitions
- reward signals
- semantic events
- permissions
- snapshots

```
AI Runtime
    |
Program Interface
    |
AI-Native OS
    |
Apps / Worlds / Tools
```

The runtime inhabits the OS continuously. The OS becomes an experience
platform.

## Not a neural kernel

The "OS" in this vision is not a neural network pretending to be a kernel.
It is a **deterministic computing substrate** built around a continuous
cognitive runtime:

- an AI-native shell
- event-native apps
- semantic device streams
- persistent goals
- a memory-first workspace
- programs as experience generators

Determinism, snapshots and replay — the same properties the SurvivalBox
backend guarantees today — are what make the substrate debuggable and safe
at OS scale.

## Why the MVP architecture already points there

Every design constraint in the MVP exists to keep this future reachable:

| MVP property | AI-OS consequence |
|---|---|
| Environment-agnostic runtime core | The same runtime moves from a game to a desktop to an OS |
| Universal Program interface | OS apps implement the same contract SurvivalBox does |
| NULL as a real action | A continuously-running agent must mostly *not* act |
| Deterministic seeds + snapshots | System state is checkpointable and auditable |
| Tick recording + replay | Every behavior is reproducible for debugging and training |
| Program-side rewards & events | Apps define what "good" means in their own domain |

## Defining principle

Programs do not tell the runtime exactly how to solve problems.
Programs create experiences. The runtime learns from those experiences.
That is the foundation of the entire project.
