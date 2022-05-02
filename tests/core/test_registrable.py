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


def test_registrable_with_name():
    class A(Registrable):
        pass

    @A.register(name="b")
    class B(A):
        pass

    assert A in Registrable._registry
    assert Registrable._registry[A] == {"b": B}

    class_type = A.by_name("b")
    assert class_type is B

    clazz = class_type()
    assert isinstance(clazz, B)
    assert clazz.register_name == "b"

    @A.register()
    class C(A):
        pass

    assert A in Registrable._registry
    assert Registrable._registry[A] == {"b": B, "C": C}

    class_type = A.by_name("C")
    assert class_type is C

    clazz = class_type()
    assert isinstance(clazz, C)
    assert clazz.register_name == "C"
