import pytest
from numpy.testing import assert_, assert_equal, assert_raises

from pyvrp import Model, VehicleType
from pyvrp.stop import MaxRuntime


def _complete_graph(model, locations):
    """
    Adds symmetric Manhattan-distance edges between all locations.
    """
    for frm in locations:
        for to in locations:
            if frm is not to:
                dist = int(abs(frm.x - to.x) + abs(frm.y - to.y))
                model.add_edge(frm, to, distance=dist, duration=dist)


def _route_of_client(result, data, client_idx):
    """
    Returns the vehicle type index of the route serving the given client.
    """
    for route in result.best.routes():
        visits = [act.idx for act in route if act.is_client()]
        if client_idx in visits:
            return route.vehicle_type()
    return None


def test_no_compatibility_adds_no_dimensions():
    """
    A model that uses no skills or vehicle restrictions is unchanged: no extra
    load dimensions are introduced.
    """
    m = Model()
    depot = m.add_depot(m.add_location(x=0, y=0))
    m.add_client(m.add_location(x=1, y=1))
    m.add_vehicle_type(1, start_depot=depot, end_depot=depot)

    data = m.data()
    assert_equal(data.num_load_dimensions, 0)


def test_skills_add_one_dimension_with_expected_values():
    """
    A single skill compiles to one load dimension: the requiring client has one
    unit of demand, only the skilled vehicle has capacity.
    """
    m = Model()
    depot = m.add_depot(m.add_location(x=0, y=0))
    plain = m.add_client(m.add_location(x=1, y=0))
    crane = m.add_client(m.add_location(x=2, y=0), required_skills=["crane"])

    m.add_vehicle_type(1, start_depot=depot, end_depot=depot)  # no skill
    m.add_vehicle_type(1, start_depot=depot, end_depot=depot, skills=["crane"])

    data = m.data()
    assert_equal(data.num_load_dimensions, 1)

    # Client index follows insertion order: plain=0, crane=1.
    assert_equal(data.client(0).delivery, [0])  # plain: no skill demand
    assert_equal(data.client(1).delivery, [1])  # crane: one unit of demand

    # Vehicle type 0 has no skill (capacity 0), type 1 has it (positive).
    assert_equal(data.vehicle_type(0).capacity, [0])
    assert_(data.vehicle_type(1).capacity[0] > 0)

    assert_(plain is not None and crane is not None)


def test_skill_client_is_served_by_skilled_vehicle():
    """
    Solving places a skill-requiring client on the vehicle type that has the
    skill.
    """
    m = Model()
    locs = [m.add_location(x=x, y=0) for x in range(3)]
    depot = m.add_depot(locs[0])
    m.add_client(locs[1])
    m.add_client(locs[2], required_skills=["crane"])
    m.add_vehicle_type(1, start_depot=depot, end_depot=depot)
    m.add_vehicle_type(1, start_depot=depot, end_depot=depot, skills=["crane"])
    _complete_graph(m, locs)

    res = m.solve(stop=MaxRuntime(1), display=False)
    assert_(res.is_feasible())
    # Client 1 (the crane client) must be on vehicle type 1 (the skilled one).
    assert_equal(_route_of_client(res, m.data(), 1), 1)


def test_allowed_vehicle_types_by_object():
    """
    Restricting a client to a subset of vehicle types compiles to one load
    dimension per forbidden type, and the client is served by an allowed type.
    """
    m = Model()
    locs = [m.add_location(x=x, y=0) for x in range(3)]
    depot = m.add_depot(locs[0])
    veh_a = m.add_vehicle_type(1, start_depot=depot, end_depot=depot)
    veh_b = m.add_vehicle_type(1, start_depot=depot, end_depot=depot)

    m.add_client(locs[1])
    m.add_client(locs[2], allowed_vehicle_types=[veh_b])
    _complete_graph(m, locs)

    data = m.data()
    # One restricted type (veh_a, index 0) -> one dimension.
    assert_equal(data.num_load_dimensions, 1)
    assert_equal(data.client(1).delivery, [1])  # forbids veh_a
    assert_equal(data.vehicle_type(0).capacity, [0])  # veh_a cannot carry it
    assert_(data.vehicle_type(1).capacity[0] > 0)  # veh_b can

    res = m.solve(stop=MaxRuntime(1), display=False)
    assert_(res.is_feasible())
    assert_equal(_route_of_client(res, data, 1), 1)  # served by veh_b
    assert_(veh_a is not None)


def test_allowed_vehicle_types_by_name():
    """
    Allowed vehicle types may be given by name, so clients can be added before
    their vehicle types.
    """
    m = Model()
    locs = [m.add_location(x=x, y=0) for x in range(3)]
    depot = m.add_depot(locs[0])
    m.add_client(locs[1])
    m.add_client(locs[2], allowed_vehicle_types=["van"])
    m.add_vehicle_type(1, start_depot=depot, end_depot=depot, name="truck")
    m.add_vehicle_type(1, start_depot=depot, end_depot=depot, name="van")
    _complete_graph(m, locs)

    res = m.solve(stop=MaxRuntime(1), display=False)
    assert_(res.is_feasible())
    assert_equal(_route_of_client(res, m.data(), 1), 1)  # the "van"


def test_compatibility_coexists_with_real_load_dimensions():
    """
    Compatibility dimensions are appended after existing load dimensions, which
    remain intact.
    """
    m = Model()
    depot = m.add_depot(m.add_location(x=0, y=0))
    m.add_client(
        m.add_location(x=1, y=0), delivery=5, required_skills=["cold"]
    )

    m.add_vehicle_type(1, capacity=10, start_depot=depot, end_depot=depot)
    m.add_vehicle_type(
        1, capacity=10, start_depot=depot, end_depot=depot, skills=["cold"]
    )

    data = m.data()
    assert_equal(data.num_load_dimensions, 2)  # real load + skill
    assert_equal(data.client(0).delivery, [5, 1])
    assert_equal(data.vehicle_type(0).capacity[0], 10)  # real capacity kept
    assert_equal(data.vehicle_type(0).capacity[1], 0)  # no skill
    assert_(data.vehicle_type(1).capacity[1] > 0)  # has skill


def test_multiple_skills_and_restrictions_dimension_count():
    """
    The number of appended dimensions equals the number of skills plus the
    number of forbidden vehicle types.
    """
    m = Model()
    depot = m.add_depot(m.add_location(x=0, y=0))
    veh_a = m.add_vehicle_type(
        1, start_depot=depot, end_depot=depot, skills=["a"]
    )
    m.add_vehicle_type(
        1, start_depot=depot, end_depot=depot, skills=["a", "b"]
    )

    m.add_client(m.add_location(x=1, y=0), required_skills=["a"])
    m.add_client(m.add_location(x=2, y=0), required_skills=["b"])
    m.add_client(m.add_location(x=3, y=0), allowed_vehicle_types=[veh_a])

    # Skills {a, b} -> 2 dims; one forbidden type (the second) -> 1 dim.
    assert_equal(m.data().num_load_dimensions, 3)


def test_empty_restrictions_are_unrestricted():
    """
    Passing empty skills / allowed vehicle types is the same as no restriction.
    """
    m = Model()
    depot = m.add_depot(m.add_location(x=0, y=0))
    m.add_client(
        m.add_location(x=1, y=0), required_skills=[], allowed_vehicle_types=[]
    )
    m.add_vehicle_type(1, start_depot=depot, end_depot=depot, skills=[])

    assert_equal(m.data().num_load_dimensions, 0)


def test_required_client_without_capable_vehicle_raises():
    """
    A required client that no vehicle type can serve (skill nobody has) raises.
    """
    m = Model()
    depot = m.add_depot(m.add_location(x=0, y=0))
    m.add_client(m.add_location(x=1, y=0), required_skills=["unicorn"])
    m.add_vehicle_type(1, start_depot=depot, end_depot=depot)

    with assert_raises(ValueError):
        m.data()


def test_required_client_with_conflicting_restrictions_raises():
    """
    A required client allowed only on a vehicle type that lacks its required
    skill cannot be served, and raises.
    """
    m = Model()
    depot = m.add_depot(m.add_location(x=0, y=0))
    plain = m.add_vehicle_type(1, start_depot=depot, end_depot=depot)
    m.add_vehicle_type(1, start_depot=depot, end_depot=depot, skills=["crane"])

    # Requires 'crane', but only allowed on the vehicle without it.
    m.add_client(
        m.add_location(x=1, y=0),
        required_skills=["crane"],
        allowed_vehicle_types=[plain],
    )

    with assert_raises(ValueError):
        m.data()


def test_unknown_vehicle_type_name_raises():
    """
    Referencing a vehicle type by an unknown name raises.
    """
    m = Model()
    depot = m.add_depot(m.add_location(x=0, y=0))
    m.add_client(m.add_location(x=1, y=0), allowed_vehicle_types=["ghost"])
    m.add_vehicle_type(1, start_depot=depot, end_depot=depot, name="real")

    with assert_raises(ValueError):
        m.data()


def test_allowed_vehicle_type_not_in_model_raises():
    """
    Referencing a VehicleType object that is not part of the model raises.
    """
    stray = VehicleType()
    m = Model()
    depot = m.add_depot(m.add_location(x=0, y=0))
    m.add_client(m.add_location(x=1, y=0), allowed_vehicle_types=[stray])
    m.add_vehicle_type(1, start_depot=depot, end_depot=depot)

    with assert_raises(ValueError):
        m.data()


def test_optional_client_without_capable_vehicle_is_allowed():
    """
    An *optional* client that no vehicle can serve does not raise; it simply
    will not be visited.
    """
    m = Model()
    depot = m.add_depot(m.add_location(x=0, y=0))
    m.add_client(
        m.add_location(x=1, y=0),
        prize=10,
        required=False,
        required_skills=["unicorn"],
    )
    m.add_vehicle_type(1, start_depot=depot, end_depot=depot)

    data = m.data()  # should not raise
    assert_equal(data.num_load_dimensions, 1)


@pytest.mark.parametrize("by_name", [True, False])
def test_allowed_reference_forms_compile_identically(by_name):
    """
    Referencing an allowed vehicle type by object or by name yields the same
    compiled data.
    """
    m = Model()
    depot = m.add_depot(m.add_location(x=0, y=0))
    truck = m.add_vehicle_type(
        1, start_depot=depot, end_depot=depot, name="truck"
    )
    ref = "truck" if by_name else truck
    m.add_client(m.add_location(x=1, y=0), allowed_vehicle_types=[ref])
    m.add_vehicle_type(1, start_depot=depot, end_depot=depot, name="van")

    data = m.data()
    # The client forbids the "van" (index 1), so that is the restricted dim.
    assert_equal(data.num_load_dimensions, 1)
    assert_equal(data.client(0).delivery, [1])
    assert_equal(data.vehicle_type(1).capacity, [0])  # van cannot serve it
    assert_(data.vehicle_type(0).capacity[0] > 0)  # truck can
