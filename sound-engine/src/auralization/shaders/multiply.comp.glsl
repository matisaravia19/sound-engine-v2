#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer SignalBuffer {
	float data[];
} signal;

layout(set = 0, binding = 1) readonly buffer IrBuffer {
	float data[];
} ir;

layout(push_constant) uniform Params {
	uint count;
} params;

void main() {
	uint idx = gl_GlobalInvocationID.x;
	if (idx >= params.count) {
		return;
	}

	uint base = idx * 2u;

	float ar = signal.data[base];
	float ai = signal.data[base + 1u];
	float br = ir.data[base];
	float bi = ir.data[base + 1u];

	signal.data[base] = ar * br - ai * bi;
	signal.data[base + 1u] = ar * bi + ai * br;
}
