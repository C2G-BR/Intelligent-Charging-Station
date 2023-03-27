import { Component, OnInit, ElementRef, ViewChild } from '@angular/core';
import { UntypedFormGroup, UntypedFormBuilder } from '@angular/forms';
import { Vehicle } from '../vehicle.model';
import { map, retry } from 'rxjs/operators';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { ChartType } from 'angular-google-charts';

@Component({
    selector: 'app-home',
    templateUrl: './home.component.html',
    styleUrls: ['./home.component.scss']
})

export class HomeComponent implements OnInit {

    // Declaration of Variables (Charging Station Details, Vehicles)
    algorithm: string = "ddpg";
    max_power: number = 200;
    duration_time_step: number = 1;
    positions: number = 20;
    slot_form: Boolean = false;
    api_address: string = "http://127.0.0.1:5000";
    simulate_button: Boolean = true;

    // Declaration of Variable for Chart
    chart = {
        "title": "Capacity History",
        "dynamicResize": true,
        "columnNames": ["Time", "Capacity"],
        "type": ChartType.LineChart,
        "data": [[0, 0]],
        "options": {   
            "hAxis": {
               "title": 'Time in h'
            },
            "vAxis":{
               "title": 'Capacity in kWh'
            },
            "legend": {
                "position": "none"
            },
        }
    };
    slots: { [id: number]: Vehicle; } = {
    };

    // Group all input fields of the Charging Station variables and fill them with the standard values
    changePowerStation: UntypedFormGroup = this.formBuilder.group({
        algorithm: this.algorithm,
        max_power: this.max_power,
        duration_time_step: this.duration_time_step,
        positions: this.positions,
        api_address: this.api_address
    });

    // Saves new Values to the Charging Station variables
    public powerStation() {
        this.algorithm = this.changePowerStation.value.algorithm;
        this.max_power = this.changePowerStation.value.max_power;
        this.duration_time_step = this.changePowerStation.value.duration_time_step;
        this.positions = this.changePowerStation.value.positions;
        this.api_address = this.changePowerStation.value.api_address;
        alert('Loading Station Credentials Successfully Saved');
    }

    // Group all input fields of on Slot and fill them with the standard values
    public changeVehicle: UntypedFormGroup = this.formBuilder.group({
        position: null,
        capacity_max: null,
        charging_power_min: null,
        charging_power_max: null,
        power_dissipation: null,
        capacity_actual: null,
        limit: null,
        time_steps: null,
        exp_charge: null,
    });

    // Change Car Credentials for one Slot and save them to the Array
    public slot() {
        this.slots[this.changeVehicle.value.position]["capacity_history"].push(this.changeVehicle.value.capacity_actual)
        this.slots[this.changeVehicle.value.position] = { 
            capacity_max: this.changeVehicle.value.capacity_max,
            charging_power_min: this.changeVehicle.value.charging_power_min,
            charging_power_max: this.changeVehicle.value.charging_power_max,
            power_dissipation: this.changeVehicle.value.power_dissipation,
            capacity_history: this.slots[this.changeVehicle.value.position]["capacity_history"],
            limit: this.changeVehicle.value.limit,
            time_steps: this.changeVehicle.value.time_steps,
            exp_charge: this.changeVehicle.value.exp_charge,
        };
        this.changeVehicle.setValue({
            position: null,
            capacity_max: null,
            charging_power_min: null,
            charging_power_max: null,
            power_dissipation: null,
            capacity_actual: null,
            limit: null,
            time_steps: null,
            exp_charge: null,
        });
        this.slot_form = false;
        alert('Car Credentials Saved');
    }

    // Function to sort without ASC oder DESC
    returnZero() {
        return 0;
    }

    // Function after click on an slot. Opens the Car Credentials Form and fill the fields with the actual values
    public change_vehicle(event, position) {
        if(this.slots[position]["capacity_max"] != null) {
            this.changeVehicle.setValue({
                position: position,
                capacity_max: this.slots[position]["capacity_max"],
                capacity_actual: this.slots[position]["capacity_history"][this.slots[position]["capacity_history"].length - 1],
                charging_power_min: this.slots[position]["charging_power_min"],
                charging_power_max: this.slots[position]["charging_power_max"],
                power_dissipation: this.slots[position]["power_dissipation"],
                limit: this.slots[position]["limit"],
                time_steps: this.slots[position]["time_steps"],
                exp_charge: this.slots[position]["exp_charge"],
            });
            var chart_values = [];
            for (let i: number = 0; i <= this.slots[position]["capacity_history"].length-1; i++) {
                chart_values[i] = [i, Number(this.slots[position]["capacity_history"][i])];
            }
            this.chart.data = chart_values;
        } else {
            this.changeVehicle.setValue({
                position: position,
                capacity_max: null,
                charging_power_min: null,
                charging_power_max: null,
                power_dissipation: 0.05,
                capacity_actual: null,
                limit: null,
                time_steps: 0,
                exp_charge: -0.05,
            });
            this.chart.data = [[0, 0]];
        }
        this.slot_form = true;
    }

    // Deletes a vehicle from the slot. Clears the Values on the array and the input fields
    public delete_vehicle(event, position) {
        this.slots[position] = { 
            capacity_max: null,
            charging_power_min: null,
            charging_power_max: null,
            power_dissipation: null,
            capacity_history: null,
            limit: null,
            time_steps: null,
            exp_charge: null,
         }
         if(this.changeVehicle.value.position == position) {
            this.slot_form = false;
            this.changeVehicle.setValue({
                position: null,
                capacity_max: null,
                charging_power_min: null,
                charging_power_max: null,
                power_dissipation: null,
                capacity_actual: null,
                limit: null,
                time_steps: null,
                exp_charge: null,
            });
         }
    }

    // Functions to call the backend for prediction. First set data Object with all necessary values
    simulate() {
        this.simulate_button = false;
        var data = {
            'algorithm': this.algorithm,
            'power_station': {
                'max_power': this.max_power,
                'duration_time_step': this.duration_time_step,
                'positions': this.positions,
            },
            'vehicles': [
            ]
        };
        for (const k in this.slots) {
            if(this.slots[k]["capacity_max"] != null) {
                var slot = {
                    'position': Number(k),
                    'capacity_max': this.slots[k]["capacity_max"],
                    'charging_power_min': this.slots[k]["charging_power_min"],
                    'charging_power_max': this.slots[k]["charging_power_max"],
                    'power_dissipation': this.slots[k]["power_dissipation"],
                    'capacity_history': this.slots[k]["capacity_history"],
                    'limit': this.slots[k]["limit"],
                    'time_steps': this.slots[k]["time_steps"],
                    'exp_charge': this.slots[k]["exp_charge"]
                };
                data["vehicles"].push(slot);
            }
        }
        var result = this.httpClient.post(this.api_address + '/simulate', data).pipe(
            retry(2)
        ).subscribe((data: any) => {
            this.algorithm = data["algorithm"];
            this.max_power = data["power_station"]["max_power"];
            this.duration_time_step = data["power_station"]["duration_time_step"];
            this.positions = data["power_station"]["positions"];
            for (let i: number = 0; i <= data["vehicles"].length-1; i++) {
                var vehicle = data["vehicles"][i]
                var position = vehicle["position"]
                this.slots[position]["capacity_max"] = vehicle["capacity_max"];
                this.slots[position]["charging_power_max"] = vehicle["charging_power_max"];
                this.slots[position]["charging_power_min"] = vehicle["charging_power_min"];
                this.slots[position]["power_dissipation"] = vehicle["power_dissipation"];
                this.slots[position]["capacity_history"] = vehicle["capacity_history"];
                this.slots[position]["limit"] = vehicle["limit"];
                this.slots[position]["time_steps"] = vehicle["time_steps"];
                this.slots[position]["exp_charge"] = vehicle["exp_charge"];
            }
        });
        this.slot_form = false;
        this.simulate_button = true;
    }

    public getOptions() {
        const headers = new HttpHeaders({
          'Content-Type': 'application/json'
        });
        let options = { headers: headers };
        return options;
    }

    // Functions to create demo data of Cars in the Slots
    create_demo_data() {
        var result = this.httpClient.get(this.api_address + '/generate').pipe(
            retry(2),
        ).subscribe((data: any) => {
            this.algorithm = data["algorithm"];
            this.max_power = data["power_station"]["max_power"];
            this.duration_time_step = data["power_station"]["duration_time_step"];
            this.positions = data["power_station"]["positions"];
            for (let i: number = 0; i <= data["vehicles"].length-1; i++) {
                var vehicle = data["vehicles"][i]
                var position = vehicle["position"]
                this.slots[position]["capacity_max"] = vehicle["capacity_max"];
                this.slots[position]["charging_power_max"] = vehicle["charging_power_max"];
                this.slots[position]["charging_power_min"] = vehicle["charging_power_min"];
                this.slots[position]["power_dissipation"] = vehicle["power_dissipation"];
                this.slots[position]["capacity_history"] = vehicle["capacity_history"];
                this.slots[position]["limit"] = vehicle["limit"];
                this.slots[position]["time_steps"] = vehicle["time_steps"];
                this.slots[position]["exp_charge"] = vehicle["exp_charge"];
            }
        });
    }

    constructor(
        private formBuilder: UntypedFormBuilder,
        private httpClient: HttpClient,
        // private IcsServiceService: IcsServiceService
    ) { }

    // Init Funciton, creates the number of Slots according to the variable
    ngOnInit() {
        for (let i: number = 0; i <= this.positions-1; i++) {
            this.slots[Number(i)] = { 
                capacity_max: null,
                charging_power_min: null,
                charging_power_max: null,
                power_dissipation: null,
                capacity_history: [],
                limit: null,
                time_steps: null,
                exp_charge: null,
             }
        }
    }
}
