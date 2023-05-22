# script to generate target unitaries for pulses in parent directory
import qutip as qt
import numpy as np
from qutip.fileio import qsave
from typing import Callable


def CZ() -> qt.Qobj:
    CZ = np.diag([1, 1, 1, -1])
    return qt.Qobj(CZ, dims=[[2, 2], [2, 2]])


funcs: dict[str, Callable[None, qt.Qobj]] = {
    'CZ': CZ
}


if __name__ == '__main__':
    for lbl, func in funcs.items():
        unitary = func()
        qsave(unitary, lbl)
