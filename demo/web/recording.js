// The live preview and submitted path share these exact fixed-rate samples.
export class Recording {
 constructor(pose){this.reset(pose)}
 reset(pose){this.pose=[...pose];this.start=[...pose];this.poses=[];this.active=false;this.replay=false;this.accumulator=0;this.replayIndex=0}
 begin(){this.replay=false;if(this.poses.length<720)this.active=true}
 pause(){this.active=false;this.accumulator=0}
 advance(dt,action,speed,rotation){
  if(!this.active)return;
  this.accumulator+=dt;
  while(this.accumulator>=1/24&&this.poses.length<720){
   this.accumulator-=1/24;
   this.poses.push([...this.pose]);
   const a=action,n=Math.max(1,Math.hypot(a.forward,a.right)),yaw=this.pose[3];
   this.pose[0]+=speed/24*(Math.cos(yaw)*a.right+Math.sin(yaw)*a.forward)/n;
   this.pose[1]+=speed/24*(-Math.sin(yaw)*a.right+Math.cos(yaw)*a.forward)/n;
   this.pose[3]+=a.yaw*rotation*Math.PI/180/24;
   this.pose[4]=Math.max(-Math.PI*.44,Math.min(Math.PI*.44,this.pose[4]+a.pitch*rotation*Math.PI/180/24));
  }
  if(this.poses.length===720)this.pause();
 }
 snapshot(){return {fps:24,poses:this.poses.map(p=>[...p]),end_pose:[...this.pose]}}
}
