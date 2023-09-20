function struct_array = set_field_GSD(struct_array, field_name, gsd_result)

SENSOR_POSITION = getenv_string('SENSOR_POSITION', 'LowerBack');
SENSOR_UNIT = getenv_string('SENSOR', 'SU');

SENSOR_POSITION_OUTPUT_NAME = getenv_string('SENSOR_POSITION_OUTPUT_NAME', SENSOR_POSITION);
SENSOR_UNIT_OUTPUT_NAME = getenv_string('SENSOR_UNIT_OUTPUT_NAME', SENSOR_UNIT);

names_split = strsplit(field_name,'.');
names_split = [names_split, SENSOR_UNIT_OUTPUT_NAME, SENSOR_POSITION_OUTPUT_NAME, 'GSD'];
struct_array = setfield(struct_array,names_split{1:end},gsd_result);
