# 2026-02-26T20:57:54.739790200
import vitis

client = vitis.create_client()
client.set_workspace(path="synthesis_bundle")

comp = client.create_hls_component(name = "hls_component",cfg_file = ["hls_config.cfg"],template = "empty_hls_component")

comp = client.get_component(name="hls_component")
comp.run(operation="SYNTHESIS")

vitis.dispose()

