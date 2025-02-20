from typing import List, Iterable, Tuple, Iterator
import numpy as np
import torch
from torch.utils.data import Sampler

class KShotTaskSampler(Sampler):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 episodes_per_epoch: int,
                 n: int,
                 k: int,
                 min_queries: int,
                 max_queries: int,
                 num_tasks: int = 1,
                 fixed_tasks: List[Iterable[int]] = None):
        super().__init__(dataset)
        if num_tasks < 1:
            raise ValueError('num_tasks must be >= 1.')
        if min_queries < 0 or max_queries < min_queries:
            raise ValueError('Invalid query range.')

        self.dataset = dataset
        self.episodes_per_epoch = episodes_per_epoch
        self.n = n
        self.k = k
        self.min_queries = min_queries
        self.max_queries = max_queries
        self.num_tasks = num_tasks
        self.fixed_tasks = fixed_tasks
        self.i_task = 0

    def __len__(self) -> int:
        return self.episodes_per_epoch

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray, List[str]]]:
        for _ in range(self.episodes_per_epoch):
            support_set, query_set = [], []
            for task in range(self.num_tasks):
                if self.fixed_tasks is None:
                    # Randomly sample classes
                    episode_classes = np.random.choice(
                        self.dataset.df['class_id'].unique(),
                        size=self.k,
                        replace=False
                    )
                else:
                    # Use fixed tasks
                    episode_classes = self.fixed_tasks[self.i_task % len(self.fixed_tasks)]
                    self.i_task += 1

                df = self.dataset.df[self.dataset.df['class_id'].isin(episode_classes)]
                support_k = {k: None for k in episode_classes}
                episode_labels = list(episode_classes)

                for k in episode_classes:
                    # Sample support set
                    support = df[df['class_id'] == k].sample(self.n, replace=False)
                    support_k[k] = support
                    support_set.extend([{"text": row['text'], "label": row['class_name']} for _, row in support.iterrows()])

                for k in episode_classes:
                    # Randomize query count within range
                    query_count = np.random.randint(self.min_queries, self.max_queries + 1)
                    query_candidates = df[
                        (df['class_id'] == k) &
                        (~df['id'].isin(support_k[k]['id']))
                    ]
                    if len(query_candidates) < query_count:
                        raise ValueError(f"Not enough samples for class {k} to support {query_count} queries.")
                    query = query_candidates.sample(query_count, replace=False)
                    query_set.extend([{"text": row['text'], "label": row['class_name']} for _, row in query.iterrows()])

            # Convert to array or pad/truncate as necessary for network compatibility
            support_array = np.array(support_set)
            query_array = np.array(query_set)
            yield support_array, query_array, episode_labels
