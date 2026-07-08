from __future__ import annotations

from typing import TYPE_CHECKING, Sequence
from warnings import warn

import numpy as np

from pyvrp._pyvrp import (
    Client,
    ClientGroup,
    Depot,
    Location,
    ProblemData,
    Solution,
    VehicleType,
)
from pyvrp.constants import MAX_VALUE
from pyvrp.exceptions import ScalingWarning
from pyvrp.solve import SolveParams, solve

if TYPE_CHECKING:
    from pyvrp.Result import Result
    from pyvrp.stop import StoppingCriterion


class Edge:
    """
    Stores an edge connecting two locations.

    Raises
    ------
    ValueError
        When either distance or duration is a negative value, or when self
        loops have nonzero distance or duration values.
    """

    __slots__ = ["distance", "duration", "frm", "to"]

    def __init__(
        self,
        frm: Location,
        to: Location,
        distance: int,
        duration: int,
    ):
        if distance < 0 or duration < 0:
            raise ValueError("Cannot have negative edge distance or duration.")

        if id(frm) == id(to) and (distance != 0 or duration != 0):
            raise ValueError("A self loop must have 0 distance and duration.")

        if max(distance, duration) > MAX_VALUE:
            msg = """
            The given distance or duration value is very large. This may impact
            numerical stability. Consider rescaling your input data.
            """
            warn(msg, ScalingWarning)

        self.frm = frm
        self.to = to
        self.distance = distance
        self.duration = duration


class Profile:
    """
    Stores a routing profile.

    A routing profile is a collection of edges with distance and duration
    attributes that together define a complete distance and duration matrix.
    These can be used to model, for example, the road uses of different types
    of vehicles, like trucks, cars, or bicyclists. Each
    :class:`~pyvrp._pyvrp.VehicleType` is associated with a routing profile.
    """

    edges: list[Edge]
    name: str

    def __init__(self, *, name: str = ""):
        self.edges = []
        self.name = name

    def add_edge(
        self,
        frm: Location,
        to: Location,
        distance: int,
        duration: int = 0,
    ) -> Edge:
        """
        Adds a new edge to this routing profile.
        """
        edge = Edge(frm, to, distance, duration)
        self.edges.append(edge)
        return edge

    def __str__(self) -> str:
        return self.name


class Model:
    """
    A simple interface for modelling vehicle routing problems with PyVRP.
    """

    def __init__(self) -> None:
        self._locations: list[Location] = []
        self._clients: list[Client] = []
        self._depots: list[Depot] = []
        self._edges: list[Edge] = []
        self._groups: list[ClientGroup] = []
        self._profiles: list[Profile] = []
        self._vehicle_types: list[VehicleType] = []

        # Vehicle-client compatibility metadata, index-aligned with the
        # ``_clients`` and ``_vehicle_types`` lists above. These are compiled
        # into additional load dimensions when building the problem data; see
        # ``_apply_compatibility``. The C++ Client/VehicleType objects cannot
        # store this, so it is kept here on the Python side.
        self._vehicle_skills: list[list[str]] = []
        self._client_required_skills: list[list[str]] = []
        self._client_allowed_vehicles: list[list[VehicleType | str]] = []

    @property
    def clients(self) -> list[Client]:
        """
        Returns all clients currently in the model.
        """
        return self._clients

    @property
    def depots(self) -> list[Depot]:
        """
        Returns all depots currently in the model.
        """
        return self._depots

    @property
    def locations(self) -> list[Location]:
        """
        Returns all locations in the current model.
        """
        return self._locations

    @property
    def groups(self) -> list[ClientGroup]:
        """
        Returns all client groups currently in the model.
        """
        return self._groups

    @property
    def profiles(self) -> list[Profile]:
        """
        Returns all routing profiles currently in the model.
        """
        return self._profiles

    @property
    def vehicle_types(self) -> list[VehicleType]:
        """
        Returns the vehicle types in the current model. The routes of the
        solution returned by :meth:`~solve` have a property
        :meth:`~pyvrp._pyvrp.Route.vehicle_type()` that can be used to index
        these vehicle types.
        """
        return self._vehicle_types

    @classmethod
    def from_data(cls, data: ProblemData) -> "Model":
        """
        Constructs a model instance from the given data.

        .. tip::
           Only use this method if you intend to change the data using the
           model interface. If you only want to solve the given data instance,
           it is faster to directly call :meth:`~pyvrp.solve.solve`.

        Parameters
        ----------
        data
            Problem data to feed into the model.

        Returns
        -------
        Model
            A model instance representing the given data.
        """
        locs = data.locations()
        depots = data.depots()
        clients = data.clients()

        profiles = [Profile() for _ in range(data.num_profiles)]
        for idx, profile in enumerate(profiles):
            distances = data.distance_matrix(profile=idx)
            durations = data.duration_matrix(profile=idx)
            profile.edges = [
                Edge(
                    frm=locs[frm],
                    to=locs[to],
                    distance=distances[frm, to],
                    duration=durations[frm, to],
                )
                for frm in range(data.num_locations)
                for to in range(data.num_locations)
            ]

        self = Model()
        self._locations = locs
        self._clients = clients
        self._depots = depots
        self._groups = data.groups()
        self._profiles = profiles
        self._vehicle_types = data.vehicle_types()

        return self

    def add_location(
        self,
        x: float,
        y: float,
        *,
        name: str = "",
    ) -> Location:
        """
        Adds a location with the given attributes to the model. Returns the
        created :class:`~pyvrp._pyvrp.Location` instance.
        """
        loc = Location(
            x=x,
            y=y,
            name=name,
        )

        self._locations.append(loc)
        return loc

    def add_client(
        self,
        location: Location,
        delivery: int | list[int] = [],
        pickup: int | list[int] = [],
        service_duration: int = 0,
        tw_early: int = 0,
        tw_late: int = np.iinfo(np.int64).max,
        release_time: int = 0,
        prize: int = 0,
        required: bool = True,
        group: ClientGroup | None = None,
        *,
        name: str = "",
        required_skills: list[str] = [],
        allowed_vehicle_types: list[VehicleType | str] = [],
    ) -> Client:
        """
        Adds a client with the given attributes to the model. Returns the
        created :class:`~pyvrp._pyvrp.Client` instance.

        The ``required_skills`` and ``allowed_vehicle_types`` arguments
        restrict which vehicle types may serve this client. These are compiled
        into additional load dimensions when the problem data is built, which
        makes the restriction a hard constraint (an incompatible assignment is
        infeasible). See :meth:`~add_vehicle_type` for the vehicle side.

        Parameters
        ----------
        required_skills
            Skills this client requires. Only vehicle types that have *all* of
            these skills (see :meth:`~add_vehicle_type`) may serve this client.
            Empty by default, meaning no skill requirement.
        allowed_vehicle_types
            The vehicle types allowed to serve this client, given as
            :class:`~pyvrp._pyvrp.VehicleType` instances or their names. Empty
            by default, meaning any vehicle type may serve this client.

        Raises
        ------
        ValueError
            When ``group`` is not ``None``, and the given ``group`` is not part
            of this model instance, or when a required client is being added to
            a mutually exclusive client group.
        """
        if group is None:
            group_idx = None
        elif (idx := _idx_by_id(group, self._groups)) is not None:
            group_idx = idx
        else:
            raise ValueError("The given group is not in this model instance.")

        if (location_idx := _idx_by_id(location, self._locations)) is None:
            msg = "The given location is not in this model instance."
            raise ValueError(msg)

        if required and group is not None and group.mutually_exclusive:
            # Required clients cannot be part of a mutually exclusive client
            # group, since then there's nothing to decide about.
            raise ValueError("Required client in mutually exclusive group.")

        client = Client(
            location=location_idx,
            delivery=[delivery] if isinstance(delivery, int) else delivery,
            pickup=[pickup] if isinstance(pickup, int) else pickup,
            service_duration=service_duration,
            tw_early=tw_early,
            tw_late=tw_late,
            release_time=release_time,
            prize=prize,
            required=required,
            group=group_idx,
            name=name,
        )

        if group_idx is not None:
            client_idx = len(self._clients)
            self._groups[group_idx].add_client(client_idx)

        self._clients.append(client)
        self._client_required_skills.append(list(required_skills))
        self._client_allowed_vehicles.append(list(allowed_vehicle_types))
        return client

    def add_client_group(
        self, required: bool = True, *, name: str = ""
    ) -> ClientGroup:
        """
        Adds a new, possibly optional, client group to the model. Returns the
        created group.
        """
        group = ClientGroup(required=required, name=name)
        self._groups.append(group)
        return group

    def add_depot(
        self,
        location: Location,
        tw_early: int = 0,
        tw_late: int = np.iinfo(np.int64).max,
        service_duration: int = 0,
        *,
        name: str = "",
    ) -> Depot:
        """
        Adds a depot with the given attributes to the model. Returns the
        created :class:`~pyvrp._pyvrp.Depot` instance.
        """
        if (location_idx := _idx_by_id(location, self._locations)) is None:
            msg = "The given location is not in this model instance."
            raise ValueError(msg)

        depot = Depot(
            location=location_idx,
            tw_early=tw_early,
            tw_late=tw_late,
            service_duration=service_duration,
            name=name,
        )

        self._depots.append(depot)
        return depot

    def add_edge(
        self,
        frm: Location,
        to: Location,
        distance: int,
        duration: int = 0,
        profile: Profile | None = None,
    ) -> Edge:
        """
        Adds an edge :math:`(i, j)` between ``frm`` (:math:`i`) and ``to``
        (:math:`j`). The edge can be given distance and duration attributes.
        Distance is required, but the default duration is zero. Returns the
        created edge.

        .. note::

           If ``profile`` is not provided, the edge is a base edge that will be
           set for all profiles in the model. Any profile-specific edge takes
           precedence over a base edge with the same ``frm`` and ``to``
           locations.

        .. note::

           If called repeatedly with the same ``frm``, ``to``, and ``profile``
           arguments, only the edge constructed last is used. PyVRP does not
           support multigraphs.
        """
        if profile is not None:
            return profile.add_edge(frm, to, distance, duration)

        edge = Edge(frm=frm, to=to, distance=distance, duration=duration)
        self._edges.append(edge)
        return edge

    def add_profile(self, *, name: str = "") -> Profile:
        """
        Adds a new routing profile to the model.
        """
        profile = Profile(name=name)
        self._profiles.append(profile)
        return profile

    def add_vehicle_type(
        self,
        num_available: int = 1,
        capacity: int | list[int] = [],
        start_depot: Depot | None = None,
        end_depot: Depot | None = None,
        fixed_cost: int = 0,
        tw_early: int = 0,
        tw_late: int = np.iinfo(np.int64).max,
        shift_duration: int = np.iinfo(np.int64).max,
        max_distance: int = np.iinfo(np.int64).max,
        unit_distance_cost: int = 1,
        unit_duration_cost: int = 0,
        profile: Profile | None = None,
        start_late: int | None = None,
        initial_load: int | list[int] = [],
        reload_depots: list[Depot] = [],
        max_reloads: int = np.iinfo(np.uint64).max,
        max_overtime: int = 0,
        unit_overtime_cost: int = 0,
        *,
        name: str = "",
        skills: list[str] = [],
    ) -> VehicleType:
        """
        Adds a vehicle type with the given attributes to the model. Returns the
        created :class:`~pyvrp._pyvrp.VehicleType` instance.

        .. note::

           The vehicle type is assigned to the first depot if no depot
           information is provided.

        Parameters
        ----------
        skills
            Skills that vehicles of this type have. A client that requires
            skills (see :meth:`~add_client`) may only be served by vehicle
            types that have all of the client's required skills. Empty by
            default.

        Raises
        ------
        ValueError
            When the given ``depot`` or ``profile`` arguments are not in this
            model instance.
        """
        if start_depot is None:
            start_idx = 0
        elif (idx := _idx_by_id(start_depot, self._depots)) is not None:
            start_idx = idx
        else:
            raise ValueError("The given start depot is not in this model.")

        if end_depot is None:
            end_idx = 0
        elif (idx := _idx_by_id(end_depot, self._depots)) is not None:
            end_idx = idx
        else:
            raise ValueError("The given end depot is not in this model.")

        if profile is None:
            profile_idx = 0
        elif (idx := _idx_by_id(profile, self._profiles)) is not None:
            profile_idx = idx
        else:
            raise ValueError("The given profile is not in this model.")

        reloads: list[int] = []
        for depot in reload_depots:
            depot_idx = _idx_by_id(depot, self._depots)
            if depot_idx is not None:
                reloads.append(depot_idx)
            else:
                msg = "The given reload depot is not in this model."
                raise ValueError(msg)

        init_load = initial_load
        if isinstance(init_load, int):
            init_load = [init_load]

        vehicle_type = VehicleType(
            num_available=num_available,
            capacity=[capacity] if isinstance(capacity, int) else capacity,
            start_depot=start_idx,
            end_depot=end_idx,
            fixed_cost=fixed_cost,
            tw_early=tw_early,
            tw_late=tw_late,
            shift_duration=shift_duration,
            max_distance=max_distance,
            unit_distance_cost=unit_distance_cost,
            unit_duration_cost=unit_duration_cost,
            profile=profile_idx,
            start_late=start_late,
            initial_load=init_load,
            reload_depots=reloads,
            max_reloads=max_reloads,
            max_overtime=max_overtime,
            unit_overtime_cost=unit_overtime_cost,
            name=name,
        )

        self._vehicle_types.append(vehicle_type)
        self._vehicle_skills.append(list(skills))
        return vehicle_type

    def data(self, missing_value: int = MAX_VALUE) -> ProblemData:
        """
        Creates and returns a :class:`~pyvrp._pyvrp.ProblemData` instance
        from this model's attributes.

        Parameters
        ----------
        missing_value
            Distance and duration value to use for missing edges. Defaults to
            :const:`~pyvrp.constants.MAX_VALUE`, a large number. Note that this
            value cannot exceed :const:`~pyvrp.constants.MAX_VALUE`.
        """
        locs = self._locations
        loc2idx = {id(loc): idx for idx, loc in enumerate(locs)}

        # First we create the base distance and duration matrices. These are
        # shared by all routing profiles.
        fill_value = min(missing_value, MAX_VALUE)
        base_distance = np.full((len(locs), len(locs)), fill_value, np.int64)
        base_duration = np.full((len(locs), len(locs)), fill_value, np.int64)
        np.fill_diagonal(base_distance, 0)
        np.fill_diagonal(base_duration, 0)

        for edge in self._edges:
            frm = loc2idx[id(edge.frm)]
            to = loc2idx[id(edge.to)]
            base_distance[frm, to] = edge.distance
            base_duration[frm, to] = edge.duration

        # Now we create the profile-specific distance and duration matrices.
        # These are based on the base matrices.
        distances = []
        durations = []
        for profile in self._profiles:
            prof_distance = base_distance.copy()
            prof_duration = base_duration.copy()

            for edge in profile.edges:
                frm = loc2idx[id(edge.frm)]
                to = loc2idx[id(edge.to)]
                prof_distance[frm, to] = edge.distance
                prof_duration[frm, to] = edge.duration

            distances.append(prof_distance)
            durations.append(prof_duration)

        # When the user has not provided any profiles, we create an implicit
        # first profile from the base matrices.
        if not self._profiles:
            distances = [base_distance]
            durations = [base_duration]

        # Compile any vehicle-client compatibility (skills / allowed vehicle
        # types) into additional load dimensions. Returns the clients and
        # vehicle types unchanged when no compatibility is used.
        clients, vehicle_types = self._apply_compatibility()

        return ProblemData(
            self._locations,
            clients,
            self._depots,
            vehicle_types,
            distances,
            durations,
            self._groups,
        )

    def _apply_compatibility(
        self,
    ) -> tuple[list[Client], list[VehicleType]]:
        """
        Compiles the vehicle-client compatibility metadata (client required
        skills and allowed vehicle types, and vehicle type skills) into extra
        load dimensions, appended after any existing load dimensions. Returns
        new client and vehicle type lists with the additional dimensions. When
        no compatibility is used, the original lists are returned unchanged.

        A client requiring a skill (or forbidding a vehicle type) is given one
        unit of "demand" in a dedicated dimension; only compatible vehicles
        have capacity in that dimension, so an incompatible assignment incurs
        excess load and is thus infeasible.
        """
        clients = self._clients
        vehicles = self._vehicle_types
        num_types = len(vehicles)

        # Metadata may be shorter than the client/vehicle lists (e.g. when the
        # model was built via ``from_data``). Pad with empty defaults.
        veh_skills = self._vehicle_skills + [[]] * (
            num_types - len(self._vehicle_skills)
        )
        req_skills = self._client_required_skills + [[]] * (
            len(clients) - len(self._client_required_skills)
        )
        allowed_meta = self._client_allowed_vehicles + [[]] * (
            len(clients) - len(self._client_allowed_vehicles)
        )

        # Resolve each client's allowed vehicle-type indices. ``None`` denotes
        # an unrestricted client (the default).
        name2idx: dict[str, int] = {}
        for idx, veh in enumerate(vehicles):
            if veh.name:
                name2idx.setdefault(veh.name, idx)

        allowed_sets: list[set[int] | None] = []
        for allowed in allowed_meta:
            if not allowed:
                allowed_sets.append(None)
                continue

            idxs = set()
            for ref in allowed:
                if isinstance(ref, str):
                    if ref not in name2idx:
                        msg = f"Unknown vehicle type name '{ref}'."
                        raise ValueError(msg)
                    idxs.add(name2idx[ref])
                elif (vt_idx := _idx_by_id(ref, vehicles)) is not None:
                    idxs.add(vt_idx)
                else:
                    msg = "An allowed vehicle type is not in this model."
                    raise ValueError(msg)

            allowed_sets.append(idxs)

        skills = sorted(
            {s for skls in veh_skills for s in skls}
            | {s for req in req_skills for s in req}
        )

        # A vehicle-type dimension is only needed for types that at least one
        # client forbids.
        restricted = sorted(
            {
                t
                for allowed in allowed_sets
                if allowed is not None
                for t in range(num_types)
                if t not in allowed
            }
        )

        if not skills and not restricted:  # nothing to compile
            return clients, vehicles

        # Every required client must be servable by at least one vehicle type
        # that has all its skills and is allowed.
        for c_idx, client in enumerate(clients):
            if not client.required:
                continue

            needed = set(req_skills[c_idx])
            allowed_idxs = allowed_sets[c_idx]
            if not any(
                needed <= set(veh_skills[t])
                and (allowed_idxs is None or t in allowed_idxs)
                for t in range(num_types)
            ):
                msg = (
                    f"Required client {c_idx} cannot be served by any vehicle "
                    "type given its skills and allowed vehicle types."
                )
                raise ValueError(msg)

        big = max(len(clients), 1)  # safe upper bound on per-dimension load
        base_dim = _base_load_dim(clients, vehicles)

        new_clients = []
        for c_idx, client in enumerate(clients):
            needed = set(req_skills[c_idx])
            allowed_idxs = allowed_sets[c_idx]
            extra = [1 if s in needed else 0 for s in skills]
            extra += [
                1 if allowed_idxs is not None and t not in allowed_idxs else 0
                for t in restricted
            ]
            new_clients.append(_extend_client(client, base_dim, extra))

        new_vehicles = []
        for t_idx, veh in enumerate(vehicles):
            has = set(veh_skills[t_idx])
            extra = [big if s in has else 0 for s in skills]
            extra += [0 if t == t_idx else big for t in restricted]
            new_vehicles.append(_extend_vehicle(veh, base_dim, extra))

        return new_clients, new_vehicles

    def solve(
        self,
        stop: StoppingCriterion,
        seed: int = 0,
        collect_stats: bool = True,
        display: bool = True,
        params: SolveParams = SolveParams(),
        missing_value: int = MAX_VALUE,
        initial_solution: Solution | None = None,
    ) -> Result:
        """
        Solve this model.

        Parameters
        ----------
        stop
            Stopping criterion to use.
        seed
            Seed value to use for the random number stream. Default 0.
        collect_stats
            Whether to collect statistics about the solver's progress. Default
            ``True``.
        display
            Whether to display information about the solver progress. Default
            ``True``. Progress information is only available when
            ``collect_stats`` is also set, which it is by default.
        params
            Solver parameters to use. If not provided, a default will be used.
        missing_value
            Distance and duration value to use for missing edges. Defaults to
            :const:`~pyvrp.constants.MAX_VALUE`, a large number.
        initial_solution
            Optional solution to use as a warm start. The solver constructs a
            (possibly poor) initial solution if this argument is not provided.

        Returns
        -------
        Result
            A Result object, containing statistics (if collected) and the best
            found solution.
        """
        return solve(
            self.data(missing_value),
            stop,
            seed,
            collect_stats,
            display,
            params,
            initial_solution,
        )


def _idx_by_id(item: object, container: Sequence[object]) -> int | None:
    """
    Obtains the index of item in the container by identity rather than equality
    (as would happen with index()). This is important for various objects in
    the Model, because objects that compare equal may not be the same as the
    one intended. See #681 for a bug caused by this.
    """
    for idx, other in enumerate(container):
        if item is other:
            return idx

    return None


def _base_load_dim(
    clients: Sequence[Client],
    vehicles: Sequence[VehicleType],
) -> int:
    """
    Returns the number of existing load dimensions, taken as the largest load
    vector length among the given clients and vehicle types. Existing data is
    padded to this length before appending compatibility dimensions.
    """
    dim = 0
    for client in clients:
        dim = max(dim, len(client.delivery), len(client.pickup))
    for vehicle in vehicles:
        dim = max(dim, len(vehicle.capacity), len(vehicle.initial_load))
    return dim


def _pad(values: Sequence[int], length: int) -> list[int]:
    """
    Right-pads the given sequence with zeros to the requested length.
    """
    return list(values) + [0] * (length - len(values))


def _extend_client(
    client: Client, base_dim: int, extra_delivery: list[int]
) -> Client:
    """
    Returns a copy of the client whose load vectors are padded to ``base_dim``
    and then extended with the given extra delivery amounts (and matching zero
    pickups) for the appended compatibility dimensions.
    """
    zeros = [0] * len(extra_delivery)
    return Client(
        location=client.location,
        delivery=_pad(client.delivery, base_dim) + extra_delivery,
        pickup=_pad(client.pickup, base_dim) + zeros,
        service_duration=client.service_duration,
        tw_early=client.tw_early,
        tw_late=client.tw_late,
        release_time=client.release_time,
        prize=client.prize,
        required=client.required,
        group=client.group,
        name=client.name,
    )


def _extend_vehicle(
    vehicle: VehicleType, base_dim: int, extra_capacity: list[int]
) -> VehicleType:
    """
    Returns a copy of the vehicle type whose capacity and initial load are
    padded to ``base_dim`` and then extended with the given extra capacities
    (and matching zero initial loads) for the appended compatibility
    dimensions.
    """
    zeros = [0] * len(extra_capacity)
    return vehicle.replace(
        capacity=_pad(vehicle.capacity, base_dim) + extra_capacity,
        initial_load=_pad(vehicle.initial_load, base_dim) + zeros,
    )
