from collections import defaultdict

from pytorch_ie.core.registrable import Registrable


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

    assert A.registered_name_for_class(B) == "B"
    assert A.registered_name_for_class(clazz.__class__) == "B"

    class C(A):
        pass

    assert A.registered_name_for_class(C) is None
