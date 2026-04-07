# 🔴 URGENT FIX: Stage1.unity Scene Serialization

## Problem
**Code Change**: EvaderAgent.cs has `_maxEpisodeSeconds = 40f` (modified 04-07 7:54 PM)  
**DLL Compiled**: Assembly-CSharp.dll updated (04-07 7:54 PM)  
**Scene NOT Updated**: Stage1.unity still contains old serialized value of 25 (saved 04-06 10:46 PM)

**Result**: Inspector shows 25 instead of 40 → Actual game limits to 25 seconds → Timeout problems.

---

## Root Cause
When you change a `[SerializeField]` default in code, existing scene saves don't automatically update.  
Unity must re-serialize the scene to pick up the new default.

---

## Solution (Choose ONE)

### ✅ **Option A: Force Scene Re-save (Recommended)**

1. **Unity Editor** → Open `Assets/01. Scenes/Stage1.unity`
2. Select **EvaderAgent** object in Hierarchy
3. Look at Inspector → "Evader — Episode Settings" section
4. **Manually drag the slider or click the field**:
   - `Max Episode Seconds`: Change from **25 → 40** (or just click to refresh)
5. **Save scene**: `Ctrl+S` (or File → Save Scene)
6. Unity will re-serialize the scene with the new value
7. Verify that the field still reads **40** after save

### ✅ **Option B: Delete Local Scene Metadata**

1. Close Unity Editor completely
2. Delete these folders (they are auto-generated):
   ```
   c:\IIT_DroneLearning\IIT_DroneLearning\Library\
   c:\IIT_DroneLearning\IIT_DroneLearning\Temp\
   ```
3. Reopen Unity → Stage1.unity loads fresh
4. Unity will rebuild cache with current code defaults
5. Verify Inspector shows **40**

### ✅ **Option C: Direct YAML Edit (Advanced)**

1. Edit `Assets/01. Scenes/Stage1.unity` as text (backup first!)
2. Search for `_maxEpisodeSeconds:`
3. Change `25` → `40`
4. Save and reload in Unity

---

## Verification Checklist

After applying fix, **verify in Inspector**:

| Field | Expected |
|---|---|
| Max Episode Seconds | **40** ← Critical! |
| _goalArrivalReward | **3.0** |
| _checkpointReward | **0.1** |

---

## Why This Matters

- **25s window** on large map (radius 12m spawn, up to 50m+ goal distance) = insufficient time for obstacle navigation
- **40s window** gives drone time to explore, find path, and reach goal
- Previously seeing **timeout→0reward** was due to episode ending before goal arrival
- With **40s + -0.001 time penalty** (vs -0.003), dronegets 4x window while penalty is 3x lighter

---

## Post-Fix Training

Once verified:

1. Recompile Unity if needed (**Assets → Reimport All**)
2. Resume training: Jupyter notebook cell 5 with `RESUME=True`
3. Monitor TensorBoard:
   - Mean Reward should stabilize (less oscillation)
   - Episode Length should show meaningful exploration (500-1000 steps)
   - Fewer "instant timeout" episodes

---

**Timestamp**: Applied 2026-04-07 during stage1_v5 training (900k→1M steps)

