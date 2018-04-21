import time
import numpy as np

class DataLoader(object):
    def __init__(self, once=False, random=False, shuffle=False):
        self._once = once
        self._random = random
        self._shuffle = shuffle
        self._shuffle_database_inds()

    @property
    def images(self):
        return self._images

    @property
    def num_images(self):
        return len(self._images)

    @property
    def batch_size(self):
        return self._batch_size

    def _shuffle_database_inds(self):
        """ Randomly permute the training database """
        if self._random:
            st0 = np.random.get_state()
            millis = int(round(time.time() * 1000) % 4294967295)
            np.random.seed(millis)

        if self._shuffle:
            self._perm = np.random.permutation(np.arange(self.num_images))
        else:
            self._perm = np.arange(self.num_images)

        if self._random:
            np.random.set_state(st0)

        self._cur = 0

    def _next_minibatch_inds(self):
        if self._cur + self.batch_size >= self.num_images:
            if self._once:
                raise StopIteration()
            self._shuffle_database_inds()
        
        db_inds = self._perm[self._cur:self._cur + self.batch_size]
        self._cur += self.batch_size

        return db_inds

    def next_minibatch(self, db_inds):
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        db_inds = self._next_minibatch_inds()
        batch = self.next_minibatch(db_inds)
        return batch
