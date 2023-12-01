## Concept
This approach offers object-oriented features for handling YAML files, enhancing conciseness and flexibility. Here's how to configure these features.

### Inheritance
Keyword: `parent_path`, a string indicating the path to the parent node. Example:
```yaml
# Actual config file

data1: 'real'
parent_path: parent # Path to the parent YAML file, can be absolute or relative
```
```yaml
# Parent YAML content

data1: 'parent'
data2: 'parent'
```
Final YAML result:
```yaml

data1: 'real'
data2: 'parent'
```
Injection modifications into secondary sub-properties are not supported.

### References
Keyword: `definitions`, an object representing a collection of referenced YAML paths, suffix: `_path`. Example:
```yaml
# Actual config file
definitions:
  key1_path: 'path1.yaml'
```

```yaml
# path1.yaml

data1: 'real'
```

Final YAML result:
```yaml
key1:
  data1: 'real'
```

Note: References support cascading, e.g.:
```yaml
# Actual config file
definitions:
  key1_path: 'path1.yaml'
```
```yaml
# path1.yaml
data1: 'real'
definitions:
  key2_path: 'path2.yaml'
```
```yaml
# path2.yaml
data2: 'path2'
```
Final result:
```yaml
key1:
  data1: 'real'
  key2:
    data2: 'path2'
```

References and inheritance can be used together.

## Usage
Inheritance and references in YAML enable quick, common configuration definitions.

A complete example is in [config_demo.yaml](config_demo.yaml).

## Extensions
- [ ] Support for injecting modifications into multiple sub-properties
- [ ] Separate `yaml_read` into an independent package for configuration inclusion.