#pragma once

#include "map.hh"
#include <algorithm>
#include <array>
#include <vector>
#include <tuple>

#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

class vehicle_model{
    double max_speed = 20.0;
    double min_speed = 0.0;
    pt_map * map;

public:
    using state = array<double, 2>;
    using control = array<double, 1>;

    vehicle_model(pt_map * input_map) : map(input_map) {
        cout << "Initialize vehicle_model" << endl;
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
        double violation = max<double>(0, x[1] - max_speed);
        total_cost += 1e5 * violation * violation;

        violation = max<double>(min_speed - x[1], 0.0);
        total_cost += 1e5 * violation * violation;

        return total_cost;
    }
};
