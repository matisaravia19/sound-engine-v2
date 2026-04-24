#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer SignalBuffer {
	float data[];
} signal;

layout(set = 0, binding = 1) readonly buffer IrBuffer {
	float data[];
} ir;

void main() {
	uint idx = gl_GlobalInvocationID.x;
	signal.data[idx] *= ir.data[idx];
}
