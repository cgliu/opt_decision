#pragma once

#include "map.hh"

#include <array>
#include <iostream>
#include <memory>

class vehicle_model{
    double max_speed = 20.0;
    double min_speed = 0.0;
    std::shared_ptr<pt_map> map;

public:
    using state = std::array<double, 2>;
    using control = std::array<double, 1>;

    vehicle_model(std::shared_ptr<pt_map> input_map) : map(input_map) {
    }

    state dynamics(state x, control u, double dt)
    {
        x[0] += x[1] * dt + 0.5 * u[0] * dt * dt;
        x[1] += u[0] * dt;
        return x;
    }

    double collision(const state & x, double time)
    {
        return map->get(x[0], time);
    }

    double cost(const state & x, const control & u, double time)
    {
        double total_cost = 0;

        // collision cost
        total_cost += 1e5 * collision(x, time);

        // speed reward
        total_cost += 1.0 * (max_speed - x[1]);

        // odom reward
        total_cost += 1.0 * (max_speed * 10.0 - x[0]);

        // control effort
        total_cost += 1.0 * u[0] * u[0];

        // speed limits
        double violation = std::max<double>(0, x[1] - max_speed);
        total_cost += 1e5 * violation * violation;

        violation = std::max<double>(min_speed - x[1], 0.0);
        total_cost += 1e5 * violation * violation;

        return total_cost;
    }
};
