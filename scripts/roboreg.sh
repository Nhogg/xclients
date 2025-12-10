uv run roboreg.py --cams 0  --show --urdf xarm7_standalone.urdf  \
    --da.host carina.cs.luc.edu --da.port 8002 \
    --sam.host carina.cs.luc.edu --sam.port 8003 \
    --sam.prompt 'robot' --sam.confidence 0.3 \
    --roboreg.host carina.cs.luc.edu --roboreg.port 8004 \
    $@

