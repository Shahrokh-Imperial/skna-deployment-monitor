from skna_framework.core import persistence_gate

def test_two_update_gate():
    assert list(persistence_gate([1,1,0],2))==[False,True,False]
