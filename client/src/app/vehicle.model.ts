export interface Vehicle {
    capacity_max: number,
    charging_power_min: number,
    charging_power_max: number,
    power_dissipation: number,
    capacity_history: Array<number>,
    limit: number,
    time_steps: number,
    exp_charge: number,
}
