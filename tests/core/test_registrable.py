from collections import defaultdict

from numpy import isin

from pytorch_ie.core.registerable import Registrable


def test_registrable():
    class A(Registrable):
        pass

    @A.register()
    class B(A):
        pass

    assert A in Registrable._registry
    assert Registrable._registry[A] == {"B": B}

    class_type = A.by_name("B")
    assert class_type is B

    clazz = class_type()
    assert isinstance(clazz, B)
