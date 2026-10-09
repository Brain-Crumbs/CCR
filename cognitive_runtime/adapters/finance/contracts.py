"""Bridge validated snapshot identities into the existing neutral Factory types."""
from cognitive_runtime.training.model_factory.task_contracts import TaskDataContract, TaskIdentity
from .schemas import ContextSnapshot


def snapshot_data_contract(identity: TaskIdentity, corpus_id: str, *, train=(), validation=(), test=()):
    """Describe explicit split membership without implementing split generation."""
    if type(identity) is not TaskIdentity or identity.domain != 'market':
        raise ValueError('market TaskIdentity required')
    splits = {'train': tuple(train), 'validation': tuple(validation), 'test': tuple(test)}
    snapshots = [snapshot for values in splits.values() for snapshot in values]
    if not snapshots or any(type(s) is not ContextSnapshot for s in snapshots):
        raise TypeError('nonempty exact ContextSnapshot corpus required')
    if len({s.record_id for s in snapshots}) != len(snapshots):
        raise ValueError('snapshot identities overlap within/across splits')
    if len({s.feature_schema for s in snapshots}) != 1:
        raise ValueError('feature schemas differ')
    return TaskDataContract(identity.to_dict(), corpus_id, snapshots[0].feature_schema,
                            {name: [{'id': s.record_id, 'sha256': s.hash} for s in values] for name, values in splits.items()})
