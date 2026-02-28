# 2026-02-28T15:06:45.009335600
import vitis

client = vitis.create_client()
client.set_workspace(path="synthesis_bundle")

comp = client.get_component(name="hls_component")
comp.run(operation="SYNTHESIS")

vitis.dispose()

